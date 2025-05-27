import { NextResponse } from "next/server"
import { getServerSession } from "next-auth/next"
import { authOptions } from "../../auth/[...nextauth]/route"
import fs from "fs"
import path from "path"
import { savePaperRecommendation, getPaperRecommendations } from "@/lib/actions/paper-recommendation"
import prisma from "@/lib/db"

// Define the session type to include user ID
interface SessionUser {
  id: string;
  name?: string | null;
  email?: string | null;
  image?: string | null;
}

interface Session {
  user: SessionUser;
}

// Function to log to console with timestamp
function logWithTimestamp(message: string) {
  const timestamp = new Date().toISOString()
  console.log(`${timestamp} - ${message}`)
}

export async function POST(req: Request) {
  try {
    // Get the repository name from the request body
    const { repoName, forceRefresh = true } = await req.json()

    if (!repoName) {
      return NextResponse.json(
        { error: "Repository name is required" },
        { status: 400 }
      )
    }
    
    logWithTimestamp(`Received recommendation request for repository: ${repoName}`)
    
    // Get the user's session to retrieve the GitHub access token
    const session = await getServerSession(authOptions) as Session | null
    if (!session || !session.user) {
      return NextResponse.json(
        { error: "User not authenticated" },
        { status: 401 }
      )
    }
    
    // Get the GitHub access token from the session or from the database
    let githubAccessToken = null
    
    // First try to get it from the JWT token if available
    if (session.user.id) {
      // Try to get the GitHub profile from the database
      const githubProfile = await prisma.githubProfile.findUnique({
        where: {
          userId: session.user.id
        }
      })
      
      if (githubProfile?.accessToken) {
        githubAccessToken = githubProfile.accessToken
        logWithTimestamp(`Retrieved GitHub access token from database for user: ${githubProfile.githubUsername}`)
      }
    }
    
    if (!githubAccessToken) {
      logWithTimestamp("No GitHub access token found for user. Authentication with GitHub repositories may fail.")
    }
    
    // Always generate fresh recommendations as requested by the user
    // We'll still save them for historical purposes, but won't return cached results

    // Call the engine server to generate recommendations
    logWithTimestamp("Calling engine server to generate recommendations")
    
    try {
      // The engine server is running on port 8000
      // Using native fetch API
      const engineResponse = await fetch("http://localhost:8000/github-recommendations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          repoName,
          github_token: githubAccessToken // Pass the GitHub access token to the backend
        }),
      })
      
      if (!engineResponse.ok) {
        const errorText = await engineResponse.text()
        logWithTimestamp(`Engine server returned error: ${engineResponse.status} ${errorText}`)
        return NextResponse.json(
          { error: `Engine server error: ${errorText}` },
          { status: engineResponse.status }
        )
      }
      
      // Parse the response from the engine server
      const recommendationData = await engineResponse.json()
      logWithTimestamp(`Received recommendation data with ${recommendationData.recommendations?.length || 0} recommendations`)

      // Save the recommendation data to the database
      const saveResult = await savePaperRecommendation(
        repoName,
        recommendationData.query,
        recommendationData
      )
      
      logWithTimestamp(`Saved recommendation data to database: ${saveResult.success ? 'success' : 'failed'}`)
      
      // Return the recommendation data to the client with additional metadata
      // Always mark as not from cache to ensure UI shows as fresh recommendations
      return NextResponse.json({
        ...recommendationData,
        fromCache: false,
        saved: saveResult.success,
        timestamp: new Date().toISOString() // Use current timestamp to show it's fresh
      })
    } catch (error) {
      logWithTimestamp(`Error calling engine server: ${error instanceof Error ? error.message : String(error)}`)
      return NextResponse.json(
        { error: "Failed to call engine server" },
        { status: 500 }
      )
    }

  } catch (error) {
    logWithTimestamp(`Error in recommendations API: ${error instanceof Error ? error.message : String(error)}`)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }

}
