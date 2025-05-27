import { NextResponse } from "next/server"
import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import prisma from "@/lib/db"

// Get saved papers for the current user
export async function GET() {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const savedPapers = await prisma.savedPaper.findMany({
      where: {
        userId: session.user.id,
      },
      include: {
        paper: true,
      },
      orderBy: {
        createdAt: "desc",
      },
    })

    return NextResponse.json(savedPapers)
  } catch (error) {
    console.error("Error fetching saved papers:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

// Save a paper for the current user
export async function POST(request: Request) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const body = await request.json()
    const { paperId, notes } = body

    if (!paperId) {
      return NextResponse.json({ error: "Paper ID is required" }, { status: 400 })
    }

    // Check if paper exists
    const paper = await prisma.paper.findUnique({
      where: {
        id: paperId,
      },
    })

    if (!paper) {
      return NextResponse.json({ error: "Paper not found" }, { status: 404 })
    }

    // Check if already saved
    const existingSavedPaper = await prisma.savedPaper.findUnique({
      where: {
        userId_paperId: {
          userId: session.user.id,
          paperId,
        },
      },
    })

    if (existingSavedPaper) {
      return NextResponse.json({ error: "Paper already saved" }, { status: 400 })
    }

    // Save the paper
    const savedPaper = await prisma.savedPaper.create({
      data: {
        userId: session.user.id,
        paperId,
        notes,
      },
      include: {
        paper: true,
      },
    })

    return NextResponse.json(savedPaper)
  } catch (error) {
    console.error("Error saving paper:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
