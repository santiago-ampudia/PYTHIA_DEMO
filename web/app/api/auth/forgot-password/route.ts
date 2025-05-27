import { NextResponse } from "next/server"
import prisma from "@/lib/db"
import { randomBytes } from "crypto"
import { sendPasswordResetEmail } from "@/lib/email"

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { email } = body

    if (!email) {
      return NextResponse.json({ error: "Email is required" }, { status: 400 })
    }

    // Check if user exists
    const user = await prisma.user.findUnique({
      where: {
        email,
      },
    })

    // Don't reveal if user exists or not for security
    if (!user || !user.password) {
      // If user doesn't exist or is using OAuth, still return success
      // This prevents email enumeration attacks
      console.log(`Password reset requested for non-existent user or OAuth user: ${email}`)
      return NextResponse.json({ success: true })
    }

    // Generate reset token
    const token = randomBytes(32).toString("hex")
    const expires = new Date(Date.now() + 3600000) // 1 hour from now

    // Save token to database
    await prisma.passwordReset.upsert({
      where: {
        userId: user.id,
      },
      update: {
        token,
        expires,
      },
      create: {
        userId: user.id,
        token,
        expires,
      },
    })

    // Send email
    const emailResult = await sendPasswordResetEmail(user.email, token)

    if (!emailResult.success) {
      console.error("Failed to send password reset email:", emailResult.error)
      // Still return success to the client for security reasons
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Error in forgot password:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
