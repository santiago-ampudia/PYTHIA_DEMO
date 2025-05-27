import { NextResponse } from "next/server"
import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import prisma from "@/lib/db"

// Add a message to a conversation
export async function POST(request: Request, { params }: { params: { id: string } }) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const conversationId = params.id
    const body = await request.json()
    const { content, isUserMessage, paperIds } = body

    if (!content) {
      return NextResponse.json({ error: "Message content is required" }, { status: 400 })
    }

    // Check if conversation exists and belongs to user
    const conversation = await prisma.conversation.findUnique({
      where: {
        id: conversationId,
        userId: session.user.id,
      },
    })

    if (!conversation) {
      return NextResponse.json({ error: "Conversation not found" }, { status: 404 })
    }

    // Create the message
    const message = await prisma.message.create({
      data: {
        conversationId,
        content,
        isUserMessage: isUserMessage ?? false,
        ...(paperIds && paperIds.length > 0
          ? {
              papers: {
                connect: paperIds.map((id: string) => ({ id })),
              },
            }
          : {}),
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

    return NextResponse.json(message)
  } catch (error) {
    console.error("Error adding message:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
