import prisma from "@/lib/db"
import { getServerSession } from "next-auth/next"

/**
 * Save a paper recommendation for a GitHub repository
 */
export async function savePaperRecommendation(
  repoName: string,
  query: string,
  recommendationData: any
) {
  try {
    // Get the current user session
    const session = await getServerSession()
    
    if (!session?.user?.email) {
      throw new Error("User not authenticated")
    }
    
    // Get the user
    const user = await prisma.user.findUnique({
      where: { email: session.user.email },
      include: {
        githubProfile: {
          include: {
            repositories: true
          }
        }
      }
    })
    
    if (!user) {
      throw new Error("User not found")
    }
    
    if (!user.githubProfile) {
      throw new Error("GitHub profile not connected")
    }
    
    // Extract the base repository name if it has a -recent-work suffix
    const baseRepoName = repoName.endsWith('-recent-work') 
      ? repoName.replace('-recent-work', '') 
      : repoName;
    
    // Find the repository using the base name
    const repository = user.githubProfile.repositories.find(
      repo => repo.name === baseRepoName
    )
    
    if (!repository) {
      throw new Error(`Repository '${baseRepoName}' not found`)
    }
    
    // Check if we already have a recommendation for this repository
    const existingRecommendation = await prisma.conversation.findFirst({
      where: {
        userId: user.id,
        title: `Paper recommendations for ${repoName}`,
        type: "github-recommendation"
      }
    })
    
    if (existingRecommendation) {
      // Update the existing conversation with a new message
      const message = await prisma.message.create({
        data: {
          conversationId: existingRecommendation.id,
          content: JSON.stringify({
            query,
            recommendations: recommendationData.recommendations
          }),
          isUserMessage: false
        }
      })
      
      // Update the conversation's updatedAt timestamp
      await prisma.conversation.update({
        where: { id: existingRecommendation.id },
        data: { updatedAt: new Date() }
      })
      
      return {
        success: true,
        message: "Paper recommendation updated",
        conversationId: existingRecommendation.id
      }
    } else {
      // Create a new conversation for this recommendation
      const conversation = await prisma.conversation.create({
        data: {
          userId: user.id,
          title: `Paper recommendations for ${repoName}`,
          type: "github-recommendation",
          messages: {
            create: {
              content: JSON.stringify({
                query,
                recommendations: recommendationData.recommendations
              }),
              isUserMessage: false
            }
          }
        }
      })
      
      return {
        success: true,
        message: "Paper recommendation saved",
        conversationId: conversation.id
      }
    }
  } catch (error) {
    console.error("Error saving paper recommendation:", error)
    return {
      success: false,
      message: error instanceof Error ? error.message : "Unknown error"
    }
  }
}

/**
 * Get paper recommendations for a GitHub repository
 */
export async function getPaperRecommendations(repoName: string) {
  try {
    // Get the current user session
    const session = await getServerSession()
    
    if (!session?.user?.email) {
      throw new Error("User not authenticated")
    }
    
    // Get the user
    const user = await prisma.user.findUnique({
      where: { email: session.user.email }
    })
    
    if (!user) {
      throw new Error("User not found")
    }
    
    // Find the conversation for this repository
    const conversation = await prisma.conversation.findFirst({
      where: {
        userId: user.id,
        title: `Paper recommendations for ${repoName}`,
        type: "github-recommendation"
      },
      include: {
        messages: {
          orderBy: { timestamp: "desc" },
          take: 1
        }
      }
    })
    
    if (!conversation || conversation.messages.length === 0) {
      return {
        success: false,
        message: "No recommendations found for this repository"
      }
    }
    
    // Parse the recommendation data from the latest message
    const latestMessage = conversation.messages[0]
    const recommendationData = JSON.parse(latestMessage.content)
    
    return {
      success: true,
      data: {
        query: recommendationData.query,
        recommendations: recommendationData.recommendations,
        timestamp: latestMessage.timestamp
      }
    }
  } catch (error) {
    console.error("Error getting paper recommendations:", error)
    return {
      success: false,
      message: error instanceof Error ? error.message : "Unknown error"
    }
  }
}

/**
 * Get all paper recommendations for the current user
 */
export async function getAllPaperRecommendations() {
  try {
    // Get the current user session
    const session = await getServerSession()
    
    if (!session?.user?.email) {
      throw new Error("User not authenticated")
    }
    
    // Get the user
    const user = await prisma.user.findUnique({
      where: { email: session.user.email }
    })
    
    if (!user) {
      throw new Error("User not found")
    }
    
    // Find all recommendation conversations
    const conversations = await prisma.conversation.findMany({
      where: {
        userId: user.id,
        type: "github-recommendation"
      },
      include: {
        messages: {
          orderBy: { timestamp: "desc" },
          take: 1
        }
      },
      orderBy: { updatedAt: "desc" }
    })
    
    // Parse the recommendations
    const recommendations = conversations.map(conversation => {
      const repoName = conversation.title.replace("Paper recommendations for ", "")
      const latestMessage = conversation.messages[0]
      const data = JSON.parse(latestMessage.content)
      
      return {
        id: conversation.id,
        repoName,
        query: data.query,
        recommendations: data.recommendations,
        timestamp: latestMessage.timestamp,
        updatedAt: conversation.updatedAt
      }
    })
    
    return {
      success: true,
      data: recommendations
    }
  } catch (error) {
    console.error("Error getting all paper recommendations:", error)
    return {
      success: false,
      message: error instanceof Error ? error.message : "Unknown error"
    }
  }
}
