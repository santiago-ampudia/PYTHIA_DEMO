import { NextResponse } from "next/server"
import prisma from "@/lib/db"
import { hash } from "bcrypt"

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { token, password } = body

    if (!token || !password) {
      return NextResponse.json({ error: "Token and password are required" }, { status: 400 })
    }

    // Find the reset token
    const passwordReset = await prisma.passwordReset.findFirst({
      where: {
        token,
        expires: {
          gt: new Date(),
        },
      },
      include: {
        user: true,
      },
    })

    if (!passwordReset) {
      return NextResponse.json({ error: "Invalid or expired token" }, { status: 400 })
    }

    // Hash the new password
    const hashedPassword = await hash(password, 10)

    // Update the user's password
    await prisma.user.update({
      where: {
        id: passwordReset.userId,
      },
      data: {
        password: hashedPassword,
      },
    })

    // Delete the reset token
    await prisma.passwordReset.delete({
      where: {
        id: passwordReset.id,
      },
    })

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Error in reset password:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
