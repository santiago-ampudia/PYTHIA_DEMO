import { NextResponse } from "next/server"
import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import prisma from "@/lib/db"

// Get conversations for the current user
export async function GET(request: Request) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const { searchParams } = new URL(request.url)
    const type = searchParams.get("type")

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

    return NextResponse.json(conversations)
  } catch (error) {
    console.error("Error fetching conversations:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

// Create a new conversation
export async function POST(request: Request) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const body = await request.json()
    const { title, type, initialMessage } = body

    if (!title || !type) {
      return NextResponse.json({ error: "Title and type are required" }, { status: 400 })
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

    return NextResponse.json(conversation)
  } catch (error) {
    console.error("Error creating conversation:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
