import { Resend } from "resend"

// Initialize Resend with your API key
const resend = new Resend(process.env.RESEND_API_KEY)

export async function sendPasswordResetEmail(email: string, token: string) {
  try {
    const resetUrl = `${process.env.NEXTAUTH_URL}/reset-password?token=${token}`

    // Log the reset URL for debugging purposes
    console.log(`
      Password reset email being sent to: ${email}
      Reset URL: ${resetUrl}
    `)

    // Always send the email using Resend
    const { data, error } = await resend.emails.send({
      from: "PaperSocial <noreply@yourdomain.com>", // Update this with your verified domain
      to: email,
      subject: "Reset your PaperSocial password",
      html: `
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto;">
          <h1 style="color: #333; text-align: center;">Reset Your Password</h1>
          <p>You requested a password reset for your PaperSocial account. Click the button below to set a new password:</p>
          <div style="text-align: center; margin: 30px 0;">
            <a href="${resetUrl}" style="background-color: #0070f3; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold;">Reset Password</a>
          </div>
          <p>If you didn't request this, you can safely ignore this email.</p>
          <p>This link will expire in 1 hour.</p>
          <hr style="border: none; border-top: 1px solid #eaeaea; margin: 20px 0;" />
          <p style="color: #666; font-size: 14px; text-align: center;">PaperSocial - Discover, discuss, and connect with research papers</p>
        </div>
      `,
    })

    if (error) {
      console.error("Error sending reset email:", error)
      return { success: false, error: error.message }
    }

    return { success: true, data }
  } catch (error) {
    console.error("Error in sendPasswordResetEmail:", error)
    return { success: false, error: "Failed to send password reset email" }
  }
}
