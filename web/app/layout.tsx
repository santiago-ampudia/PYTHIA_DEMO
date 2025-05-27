import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import "./greek-theme.css"
import { ThemeProvider } from "@/components/theme-provider"
import { AppSidebar } from "@/components/app-sidebar"
import { SidebarProvider } from "@/components/ui/sidebar"
import { Toaster } from "@/components/ui/toaster"
import { SidebarTrigger } from "@/components/ui/sidebar-trigger"
import { Providers } from "./providers"
import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "PYTHIA - Modern Research Oracle",
  description: "A modern Greek-inspired platform for research paper discovery and insights",
  generator: 'v0.dev'
}

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  const session = await getServerSession(authOptions)

  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
            <SidebarProvider>
              {session ? (
                <div className="flex min-h-screen">
                  <AppSidebar />
                  <main className="flex-1 overflow-auto">
                    <div className="fixed top-4 left-4 z-50 md:hidden">
                      <SidebarTrigger />
                    </div>
                    {children}
                  </main>
                </div>
              ) : (
                <main className="min-h-screen">{children}</main>
              )}
              <Toaster />
            </SidebarProvider>
          </ThemeProvider>
        </Providers>
      </body>
    </html>
  )
}
