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

    // Check if user has a GitHub profile
    const githubProfile = await prisma.githubProfile.findUnique({
      where: {
        userId: session.user.id,
      },
    })

    // Check if the token is still valid
    let isTokenValid = false
    if (githubProfile?.accessToken) {
      try {
        const response = await fetch("https://api.github.com/user", {
          headers: {
            Authorization: `token ${githubProfile.accessToken}`,
            Accept: "application/vnd.github.v3+json",
          },
        })
        isTokenValid = response.ok
      } catch (error) {
        console.error("Error validating GitHub token:", error)
        isTokenValid = false
      }
    }

    return NextResponse.json({
      connected: !!githubProfile && isTokenValid,
      profile:
        githubProfile && isTokenValid
          ? {
              username: githubProfile.githubUsername,
              connectedAt: githubProfile.createdAt,
            }
          : null,
    })
  } catch (error) {
    console.error("Error checking GitHub status:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
