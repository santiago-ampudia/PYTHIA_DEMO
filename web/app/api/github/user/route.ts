import { NextResponse } from "next/server"
import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import prisma from "@/lib/db"

export async function GET() {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    // Get GitHub profile
    const githubProfile = await prisma.githubProfile.findUnique({
      where: {
        userId: session.user.id,
      },
    })

    if (!githubProfile || !githubProfile.accessToken) {
      return NextResponse.json({ error: "GitHub account not connected" }, { status: 404 })
    }

    // Fetch user data from GitHub API
    const response = await fetch("https://api.github.com/user", {
      headers: {
        Authorization: `token ${githubProfile.accessToken}`,
        Accept: "application/vnd.github.v3+json",
      },
    })

    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.statusText}`)
    }

    const userData = await response.json()
    return NextResponse.json(userData)
  } catch (error) {
    console.error("Error fetching GitHub user:", error)
    return NextResponse.json({ error: "Failed to fetch GitHub user data" }, { status: 500 })
  }
}
