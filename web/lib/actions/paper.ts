"use server"

import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import prisma from "@/lib/db"
import { revalidatePath } from "next/cache"

// Save a paper for the current user
export async function savePaper(paperId: string, notes?: string) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return { error: "Unauthorized" }
    }

    // Check if paper exists
    const paper = await prisma.paper.findUnique({
      where: {
        id: paperId,
      },
    })

    if (!paper) {
      return { error: "Paper not found" }
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
      // If already saved, update notes if provided
      if (notes !== undefined) {
        const updatedSavedPaper = await prisma.savedPaper.update({
          where: {
            id: existingSavedPaper.id,
          },
          data: {
            notes,
          },
          include: {
            paper: true,
          },
        })
        revalidatePath("/personal")
        return { success: true, data: updatedSavedPaper }
      }
      return { error: "Paper already saved" }
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

    revalidatePath("/personal")
    return { success: true, data: savedPaper }
  } catch (error) {
    console.error("Error saving paper:", error)
    return { error: "Failed to save paper" }
  }
}

// Remove a saved paper
export async function removeSavedPaper(paperId: string) {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return { error: "Unauthorized" }
    }

    // Check if paper is saved
    const savedPaper = await prisma.savedPaper.findUnique({
      where: {
        userId_paperId: {
          userId: session.user.id,
          paperId,
        },
      },
    })

    if (!savedPaper) {
      return { error: "Paper not saved" }
    }

    // Delete the saved paper
    await prisma.savedPaper.delete({
      where: {
        id: savedPaper.id,
      },
    })

    revalidatePath("/personal")
    return { success: true }
  } catch (error) {
    console.error("Error removing saved paper:", error)
    return { error: "Failed to remove saved paper" }
  }
}

// Get saved papers for the current user
export async function getSavedPapers() {
  try {
    const session = await getServerSession(authOptions)

    if (!session?.user?.id) {
      return { error: "Unauthorized" }
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

    return { success: true, data: savedPapers }
  } catch (error) {
    console.error("Error fetching saved papers:", error)
    return { error: "Failed to fetch saved papers" }
  }
}
