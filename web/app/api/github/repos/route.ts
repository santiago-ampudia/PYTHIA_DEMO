import { NextResponse } from "next/server"
import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import prisma from "@/lib/db"

export async function GET() {
  try {
    const session = await getServerSession(authOptions)

    // Type assertion for session.user to include id property
    type UserWithId = {
      id: string;
      name?: string;
      email?: string;
      image?: string;
    }

    if (!session || !session.user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }
    
    const user = session.user as UserWithId
    
    if (!user.id) {
      return NextResponse.json({ error: "User ID not found" }, { status: 401 })
    }
    
    const userId = user.id

    // Get GitHub profile
    const githubProfile = await prisma.githubProfile.findUnique({
      where: {
        userId: userId,
      },
    })

    if (!githubProfile || !githubProfile.accessToken) {
      return NextResponse.json({ error: "GitHub account not connected" }, { status: 404 })
    }

    // Fetch repositories from GitHub API
    const response = await fetch("https://api.github.com/user/repos?sort=updated&per_page=100", {
      headers: {
        Authorization: `token ${githubProfile.accessToken}`,
        Accept: "application/vnd.github.v3+json",
      },
    })

    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.statusText}`)
    }

    const repos = await response.json()
    
    // Synchronize repositories with database
    try {
      // First, get existing repositories for this GitHub profile
      const existingRepos = await prisma.githubRepository.findMany({
        where: {
          githubProfileId: githubProfile.id
        }
      })
      
      // Create a map of existing repositories by name for quick lookup
      const existingRepoMap = new Map(existingRepos.map(repo => [repo.name, repo]))
      
      // Process each repository from GitHub API
      for (const repo of repos) {
        const repoName = repo.name
        
        // Check if repository already exists in database
        if (!existingRepoMap.has(repoName)) {
          // Repository doesn't exist, create it
          await prisma.githubRepository.create({
            data: {
              githubProfileId: githubProfile.id,
              name: repoName,
              description: repo.description || null,
              url: repo.html_url,
              stars: repo.stargazers_count || 0,
              forks: repo.forks_count || 0,
              language: repo.language || null
            }
          })
          console.log(`Created repository: ${repoName}`)
        } else {
          // Repository exists, update it with latest data
          const existingRepo = existingRepoMap.get(repoName);
          if (existingRepo) {
            await prisma.githubRepository.update({
              where: {
                id: existingRepo.id
              },
              data: {
                description: repo.description || null,
                url: repo.html_url,
                stars: repo.stargazers_count || 0,
                forks: repo.forks_count || 0,
                language: repo.language || null
              }
            })
          }
          console.log(`Updated repository: ${repoName}`)
        }
      }
      
      console.log('Repositories synchronized successfully')
    } catch (syncError) {
      console.error('Error synchronizing repositories:', syncError)
      // Continue anyway, as we still want to return the repositories to the client
    }
    
    return NextResponse.json(repos)
  } catch (error) {
    console.error("Error fetching GitHub repositories:", error)
    return NextResponse.json({ error: "Failed to fetch GitHub repositories" }, { status: 500 })
  }
}
