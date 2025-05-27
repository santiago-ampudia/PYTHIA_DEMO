import { NextResponse } from "next/server"
import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import prisma from "@/lib/db"

// Get user profile
export async function GET() {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const user = await prisma.user.findUnique({
      where: {
        id: session.user.id,
      },
      include: {
        interests: true,
        githubProfile: true,
      },
    })

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 })
    }

    // Don't return sensitive information
    const { password, ...userWithoutPassword } = user

    return NextResponse.json(userWithoutPassword)
  } catch (error) {
    console.error("Error fetching user profile:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

// Update user profile
export async function PUT(request: Request) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const body = await request.json()
    const { name, bio, institution, interests } = body

    // Update user
    const updatedUser = await prisma.user.update({
      where: {
        id: session.user.id,
      },
      data: {
        name,
        bio,
        institution,
      },
    })

    // Update interests if provided
    if (interests && Array.isArray(interests)) {
      // Delete existing interests
      await prisma.researchInterest.deleteMany({
        where: {
          userId: session.user.id,
        },
      })

      // Create new interests
      await Promise.all(
        interests.map((interest: string) =>
          prisma.researchInterest.create({
            data: {
              userId: session.user.id,
              name: interest,
            },
          }),
        ),
      )
    }

    // Get updated user with interests
    const user = await prisma.user.findUnique({
      where: {
        id: session.user.id,
      },
      include: {
        interests: true,
      },
    })

    // Don't return sensitive information
    const { password, ...userWithoutPassword } = user!

    return NextResponse.json(userWithoutPassword)
  } catch (error) {
    console.error("Error updating user profile:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
