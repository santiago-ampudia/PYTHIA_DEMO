"use client"

import type React from "react"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { toast } from "@/components/ui/use-toast"
import { BookMarked, ArrowLeft, AlertCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isSubmitted, setIsSubmitted] = useState(false)
  const [showDevInfo, setShowDevInfo] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      const response = await fetch("/api/auth/forgot-password", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to send reset email")
      }

      setIsSubmitted(true)

      // In development, show a button to reveal the console log info
      if (process.env.NODE_ENV !== "production") {
        setShowDevInfo(true)
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "An unexpected error occurred",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container flex min-h-screen flex-col items-center justify-center">
      <div className="flex items-center gap-2 text-3xl font-bold mb-8">
        <BookMarked className="h-8 w-8" />
        <span>PaperSocial</span>
      </div>

      <Card className="mx-auto max-w-sm">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold">Forgot Password</CardTitle>
          <CardDescription>
            {isSubmitted
              ? "Check your email for a password reset link"
              : "Enter your email address and we'll send you a link to reset your password"}
          </CardDescription>
        </CardHeader>
        {!isSubmitted ? (
          <>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="m@example.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
                <Button type="submit" className="w-full" disabled={isLoading}>
                  {isLoading ? "Sending..." : "Send Reset Link"}
                </Button>
              </form>
            </CardContent>
          </>
        ) : (
          <CardContent className="space-y-4">
            <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-md text-green-800 dark:text-green-300">
              If an account exists with that email, we've sent a password reset link.
            </div>

            {showDevInfo && process.env.NODE_ENV !== "production" && (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Development Mode</AlertTitle>
                <AlertDescription>
                  In development mode, the reset link is logged to the console instead of being sent via email.
                  <Button
                    variant="link"
                    className="p-0 h-auto"
                    onClick={() => console.log("Check the server console for the password reset link")}
                  >
                    Check server console
                  </Button>
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        )}
        <CardFooter>
          <Button variant="ghost" asChild className="w-full">
            <Link href="/login" className="flex items-center gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to Login
            </Link>
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
