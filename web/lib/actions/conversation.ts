"use server"

import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import prisma from "@/lib/db"
import { revalidatePath } from "next/cache"

// Get conversations for the current user
export async function getConversations(type?: string) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return { error: "Unauthorized" }
    }

    const whereClause = {
      userId: session.user.id,
      ...(type ? { type } : {}),
    }

    const conversations = await prisma.conversation.findMany({
      where: whereClause,
      orderBy: {
        updatedAt: "desc",
      },
      include: {
        messages: {
          orderBy: {
            timestamp: "asc",
          },
          include: {
            papers: true,
          },
        },
      },
    })

    return { success: true, data: conversations }
  } catch (error) {
    console.error("Error fetching conversations:", error)
    return { error: "Failed to fetch conversations" }
  }
}

// Create a new conversation
export async function createConversation(title: string, type: string, initialMessage?: string) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return { error: "Unauthorized" }
    }

    // Create conversation with initial message if provided
    const conversation = await prisma.conversation.create({
      data: {
        userId: session.user.id,
        title,
        type,
        ...(initialMessage
          ? {
              messages: {
                create: {
                  content: initialMessage,
                  isUserMessage: true,
                },
              },
            }
          : {}),
      },
      include: {
        messages: true,
      },
    })

    revalidatePath(`/${type}`)
    return { success: true, data: conversation }
  } catch (error) {
    console.error("Error creating conversation:", error)
    return { error: "Failed to create conversation" }
  }
}

// Delete a conversation
export async function deleteConversation(conversationId: string) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return { error: "Unauthorized" }
    }

    // Check if conversation exists and belongs to user
    const conversation = await prisma.conversation.findUnique({
      where: {
        id: conversationId,
        userId: session.user.id,
      },
    })

    if (!conversation) {
      return { error: "Conversation not found" }
    }

    // Delete the conversation (messages will be cascade deleted)
    await prisma.conversation.delete({
      where: {
        id: conversationId,
      },
    })

    revalidatePath(`/${conversation.type}`)
    return { success: true }
  } catch (error) {
    console.error("Error deleting conversation:", error)
    return { error: "Failed to delete conversation" }
  }
}

// Add a message to a conversation
export async function addMessage(conversationId: string, content: string, isUserMessage = false, paperIds?: string[]) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return { error: "Unauthorized" }
    }

    // Check if conversation exists and belongs to user
    const conversation = await prisma.conversation.findUnique({
      where: {
        id: conversationId,
        userId: session.user.id,
      },
    })

    if (!conversation) {
      return { error: "Conversation not found" }
    }

    // If we have paper IDs, check which ones exist and create the ones that don't
    let paperConnections = {}
    if (paperIds && paperIds.length > 0) {
      // Check which papers already exist
      const existingPapers = await prisma.paper.findMany({
        where: {
          id: {
            in: paperIds
          }
        },
        select: {
          id: true
        }
      })
      
      const existingPaperIds = existingPapers.map(paper => paper.id)
      const missingPaperIds = paperIds.filter(id => !existingPaperIds.includes(id))
      
      // Create missing papers with placeholder data
      if (missingPaperIds.length > 0) {
        await prisma.paper.createMany({
          data: missingPaperIds.map(id => ({
            id,
            title: `Paper ${id}`,
            authors: "Authors not available",
            year: new Date().getFullYear(),
            journal: "Unknown",
            abstract: "Abstract not available",
            url: id.startsWith("arXiv:") ? `https://arxiv.org/abs/${id.replace("arXiv:", "")}` : ""
          })),
          skipDuplicates: true
        })
      }
      
      // Now all papers exist, so we can connect them
      paperConnections = {
        papers: {
          connect: paperIds.map((id) => ({ id }))
        }
      }
    }

    // Create the message
    const message = await prisma.message.create({
      data: {
        conversationId,
        content,
        isUserMessage,
        ...paperConnections
      },
      include: {
        papers: true,
      },
    })

    // Update conversation's updatedAt
    await prisma.conversation.update({
      where: {
        id: conversationId,
      },
      data: {
        updatedAt: new Date(),
      },
    })

    revalidatePath(`/${conversation.type}`)
    return { success: true, data: message }
  } catch (error) {
    console.error("Error adding message:", error)
    return { error: "Failed to add message" }
  }
}

// Rename a conversation
export async function renameConversation(conversationId: string, newTitle: string) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return { error: "Unauthorized" }
    }

    // Check if conversation exists and belongs to user
    const conversation = await prisma.conversation.findUnique({
      where: {
        id: conversationId,
        userId: session.user.id,
      },
    })

    if (!conversation) {
      return { error: "Conversation not found" }
    }

    // Update the conversation title
    const updatedConversation = await prisma.conversation.update({
      where: {
        id: conversationId,
      },
      data: {
        title: newTitle,
      },
    })

    revalidatePath(`/${conversation.type}`)
    return { success: true, data: updatedConversation }
  } catch (error) {
    console.error("Error renaming conversation:", error)
    return { error: "Failed to rename conversation" }
  }
}
