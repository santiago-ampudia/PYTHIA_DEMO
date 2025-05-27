"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { signIn, useSession } from "next-auth/react"
import { Github, ChromeIcon as Google, Scroll } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { toast } from "@/components/ui/use-toast"

import { AuthDebug } from "@/components/auth-debug"

export default function LoginPage() {
  const router = useRouter()
  const { data: session, status } = useSession()
  const [isLoading, setIsLoading] = useState(false)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [authMethod, setAuthMethod] = useState<string | null>(null)

  // Use useEffect for navigation instead of doing it during render
  useEffect(() => {
    if (status === "authenticated") {
      router.push("/dashboard")
    }
  }, [status, router])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setAuthMethod("credentials")

    try {
      const result = await signIn("credentials", {
        email,
        password,
        redirect: false,
      })

      if (result?.error) {
        toast({
          title: "Error",
          description: "Invalid email or password",
          variant: "destructive",
        })
      } else {
        // Don't navigate here, let the useEffect handle it
        router.refresh()
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "An unexpected error occurred",
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
      await signIn("google", {
        callbackUrl: "/dashboard",
        redirect: true,
      })
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

  // If still loading, show a loading state
  if (status === "loading") {
    return (
      <div className="container flex min-h-screen flex-col items-center justify-center greek-pattern-bg">
        <div className="flex items-center gap-2 text-3xl font-bold mb-8">
          <Scroll className="h-8 w-8" />
          <span className="pythia-logo">PYTHIA</span>
        </div>
        <div className="greek-meander mx-auto w-48 my-4"></div>
        <p>Loading...</p>
      </div>
    )
  }

  // Only render the login form if not authenticated
  if (status === "unauthenticated") {
    return (
      <div className="min-h-screen w-full flex items-center justify-center bg-background p-4">
        <div className="flex flex-col items-center max-w-sm w-full">
          <div className="text-3xl font-bold mb-6 flex items-center justify-center gap-2">
            <Scroll className="h-8 w-8" />
            <span className="pythia-logo">PYTHIA</span>
          </div>

          <Card className="w-full shadow-lg">
            <CardHeader className="text-center pb-2">
              <CardTitle className="text-xl">Sign In</CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <div className="grid gap-3">
                <div className="flex gap-2">
                  <Button variant="outline" onClick={handleGithubLogin} disabled={isLoading} className="flex-1">
                    <Github className="h-4 w-4 mr-2" />
                    {authMethod === "github" && isLoading ? "..." : "GitHub"}
                  </Button>
                  <Button variant="outline" onClick={handleGoogleLogin} disabled={isLoading} className="flex-1">
                    <Google className="h-4 w-4 mr-2" />
                    {authMethod === "google" && isLoading ? "..." : "Google"}
                  </Button>
                </div>
                
                <div className="relative my-2">
                  <div className="absolute inset-0 flex items-center">
                    <span className="w-full border-t" />
                  </div>
                  <div className="relative flex justify-center text-xs">
                    <span className="bg-background px-2 text-muted-foreground">or</span>
                  </div>
                </div>
                
                <form onSubmit={handleSubmit} className="space-y-3">
                  <div>
                    <Input
                      id="email"
                      type="email"
                      placeholder="Email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </div>
                  <div>
                    <Input
                      id="password"
                      type="password"
                      placeholder="Password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                    />
                  </div>
                  <div className="text-right text-xs">
                    <Link href="/forgot-password" className="text-primary hover:underline">
                      Forgot password?
                    </Link>
                  </div>
                  <Button type="submit" className="w-full" disabled={isLoading}>
                    {authMethod === "credentials" && isLoading ? "Signing in..." : "Sign in"}
                  </Button>
                </form>
              </div>
            </CardContent>
            <CardFooter className="justify-center border-t pt-3">
              <div className="text-center text-sm">
                Don&apos;t have an account?{" "}
                <Link href="/register" className="text-primary hover:underline">
                  Sign up
                </Link>
              </div>
            </CardFooter>
          </Card>
        </div>
      </div>
    )
  }

  // This return is needed for TypeScript, but should never be reached
  // due to the useEffect redirecting authenticated users
  return null
}
