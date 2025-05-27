"use client"

import { useSession } from "next-auth/react"
import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function AuthDebug() {
  const { data: session, status } = useSession()
  const [redirectInfo, setRedirectInfo] = useState<string | null>(null)

  useEffect(() => {
    // Get redirect info from URL
    const urlParams = new URLSearchParams(window.location.search)
    const error = urlParams.get("error")
    const callbackUrl = urlParams.get("callbackUrl")

    if (error || callbackUrl) {
      setRedirectInfo(`Error: ${error}, Callback URL: ${callbackUrl}`)
    }

    // Log authentication status changes
    console.log("Auth status changed:", status)
    if (session) {
      console.log("Session:", session)
    }
  }, [session, status])

  if (status !== "loading" && process.env.NODE_ENV !== "production") {
    return (
      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Auth Debug (Development Only)</CardTitle>
        </CardHeader>
        <CardContent>
          <div>
            <p>
              <strong>Status:</strong> {status}
            </p>
            {session && (
              <>
                <p>
                  <strong>User:</strong> {session.user?.name} ({session.user?.email})
                </p>
                <p>
                  <strong>Provider:</strong> {session.user?.provider || "unknown"}
                </p>
              </>
            )}
            {redirectInfo && (
              <p>
                <strong>Redirect Info:</strong> {redirectInfo}
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    )
  }

  return null
}
