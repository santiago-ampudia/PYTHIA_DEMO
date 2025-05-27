import { NextResponse } from "next/server"
import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import prisma from "@/lib/db"

export async function POST() {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    // Delete GitHub profile and repositories
    await prisma.githubRepository.deleteMany({
      where: {
        githubProfile: {
          userId: session.user.id,
        },
      },
    })

    await prisma.githubProfile.delete({
      where: {
        userId: session.user.id,
      },
    })

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Error disconnecting GitHub account:", error)
    return NextResponse.json({ error: "Failed to disconnect GitHub account" }, { status: 500 })
  }
}
