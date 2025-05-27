"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Github, ChromeIcon as Google, BookMarked } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { toast } from "@/components/ui/use-toast"
import { signIn, useSession } from "next-auth/react"

export default function RegisterPage() {
  const router = useRouter()
  const { data: session, status } = useSession()
  const [isLoading, setIsLoading] = useState(false)
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [authMethod, setAuthMethod] = useState<string | null>(null)

  // If already authenticated, redirect to dashboard
  if (status === "authenticated") {
    router.push("/dashboard")
    return null
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setAuthMethod("credentials")

    try {
      const response = await fetch("/api/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name,
          email,
          password,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to register")
      }

      // Sign in the user after successful registration
      const result = await signIn("credentials", {
        email,
        password,
        redirect: false,
      })

      if (result?.error) {
        toast({
          title: "Registration successful",
          description: "Please sign in with your new credentials",
        })
        router.push("/login")
      } else {
        router.push("/dashboard")
        router.refresh()
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "An unexpected error occurred",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
      setAuthMethod(null)
    }
  }

  const handleGithubLogin = async () => {
    setIsLoading(true)
    setAuthMethod("github")
    try {
      await signIn("github", { callbackUrl: "/dashboard" })
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to sign in with GitHub",
        variant: "destructive",
      })
      setIsLoading(false)
      setAuthMethod(null)
    }
  }

  const handleGoogleLogin = async () => {
    setIsLoading(true)
    setAuthMethod("google")
    try {
      await signIn("google", { callbackUrl: "/dashboard" })
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to sign in with Google",
        variant: "destructive",
      })
      setIsLoading(false)
      setAuthMethod(null)
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
          <CardTitle className="text-2xl font-bold">Create an account</CardTitle>
          <CardDescription>Enter your information to create an account</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            <Button variant="outline" onClick={handleGithubLogin} disabled={isLoading} className="gap-2">
              <Github className="h-4 w-4" />
              {authMethod === "github" && isLoading ? "Signing up..." : "Sign up with GitHub"}
            </Button>
            <Button variant="outline" onClick={handleGoogleLogin} disabled={isLoading} className="gap-2">
              <Google className="h-4 w-4" />
              {authMethod === "google" && isLoading ? "Signing up..." : "Sign up with Google"}
            </Button>
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">Or continue with</span>
              </div>
            </div>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Name</Label>
                <Input
                  id="name"
                  placeholder="John Doe"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>
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
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={8}
                />
              </div>
              <Button type="submit" className="w-full" disabled={isLoading}>
                {authMethod === "credentials" && isLoading ? "Creating account..." : "Create account"}
              </Button>
            </form>
          </div>
        </CardContent>
        <CardFooter className="flex flex-col">
          <div className="text-center text-sm text-muted-foreground">
            Already have an account?{" "}
            <Link href="/login" className="text-primary underline-offset-4 hover:underline">
              Sign in
            </Link>
          </div>
        </CardFooter>
      </Card>

      <div className="mt-8 text-center text-sm text-muted-foreground">
        <p>PaperSocial - Discover, discuss, and connect with research papers</p>
      </div>
    </div>
  )
}
