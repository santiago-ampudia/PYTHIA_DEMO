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

    // First, get user's repositories
    const reposResponse = await fetch("https://api.github.com/user/repos?sort=updated&per_page=5", {
      headers: {
        Authorization: `token ${githubProfile.accessToken}`,
        Accept: "application/vnd.github.v3+json",
      },
    })

    if (!reposResponse.ok) {
      throw new Error(`GitHub API error: ${reposResponse.statusText}`)
    }

    const repos = await reposResponse.json()

    // For each repository, get recent commits
    const commitsPromises = repos.map(async (repo: any) => {
      const commitsResponse = await fetch(
        `https://api.github.com/repos/${repo.owner.login}/${repo.name}/commits?per_page=3`,
        {
          headers: {
            Authorization: `token ${githubProfile.accessToken}`,
            Accept: "application/vnd.github.v3+json",
          },
        },
      )

      if (!commitsResponse.ok) {
        console.error(`Error fetching commits for ${repo.name}: ${commitsResponse.statusText}`)
        return []
      }

      const commits = await commitsResponse.json()
      // Add repository info to each commit
      return commits.map((commit: any) => ({
        ...commit,
        repository: {
          name: repo.name,
          url: repo.html_url,
        },
      }))
    })

    const commitsArrays = await Promise.all(commitsPromises)
    // Flatten array of arrays and sort by date
    const allCommits = commitsArrays
      .flat()
      .sort((a: any, b: any) => new Date(b.commit.author.date).getTime() - new Date(a.commit.author.date).getTime())
      .slice(0, 10) // Limit to 10 most recent commits

    return NextResponse.json(allCommits)
  } catch (error) {
    console.error("Error fetching GitHub commits:", error)
    return NextResponse.json({ error: "Failed to fetch GitHub commits" }, { status: 500 })
  }
}
