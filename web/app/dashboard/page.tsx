import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import { redirect } from "next/navigation"

export default async function Dashboard() {
  const session = await getServerSession(authOptions)

  // If not authenticated, redirect to login
  if (!session) {
    redirect("/login")
  }

  return (
    <div className="flex min-h-screen flex-col greek-pattern-bg">
      <div className="flex-1">
        <section className="w-full py-8 md:py-12 lg:py-16">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl greek-header pythia-logo">
  <span className="welcome-glow">Welcome to </span>
  <span className="pythia-glow">PYTHIA</span>
</h1>
                <div className="greek-meander mx-auto w-48 my-4"></div>
                <p className="mx-auto max-w-[700px] text-gray-500 md:text-xl dark:text-gray-400 subtitle-glow">
                  The Modern Research Oracle
                </p>
              </div>

            </div>
          </div>
        </section>

        <section className="w-full py-6 md:py-10 lg:py-12 bg-muted/50">
          <div className="greek-meander mx-auto w-full my-2"></div>
          <div className="container px-4 md:px-6 greek-border">
            <div className="grid gap-6 lg:grid-cols-2 lg:gap-12 xl:grid-cols-2">
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h2 className="text-2xl font-bold tracking-tighter sm:text-3xl greek-header">Ask the Oracle</h2>
                  <p className="max-w-[600px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                    Consult the Oracle with your questions and receive highly specific and hyper-focused insights based on relevant literature.
                  </p>
                </div>
                <div>
                  <Button asChild variant="outline" size="sm" className="greek-button">
                    <Link href="/smart-answer">Ask the Oracle</Link>
                  </Button>
                </div>
              </div>
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h2 className="text-2xl font-bold tracking-tighter sm:text-3xl greek-header">Pythia's Recommendations</h2>
                  <p className="max-w-[600px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                    Receieve literature recommendations absed on whatever you are looking for. See tweets about relevant papers for you to decide what you want to read.
                  </p>
                </div>
                <div>
                  <Button asChild variant="outline" size="sm" className="greek-button">
                    <Link href="/smart-search">Pythia's Recommendations</Link>
                  </Button>
                </div>
              </div>
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h2 className="text-2xl font-bold tracking-tighter sm:text-3xl greek-header">Personal Dashboard</h2>
                  <p className="max-w-[600px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                    Edit your preferences, access settings, view recent activity, saved papers, and get personalized
                    recommendations.
                  </p>
                </div>
                <div>
                  <Button asChild variant="outline" size="sm" className="greek-button">
                    <Link href="/personal">View Dashboard</Link>
                  </Button>
                </div>
              </div>
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h2 className="text-2xl font-bold tracking-tighter sm:text-3xl greek-header">Oracle of GitHub</h2>
                  <p className="max-w-[600px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                    Connect your GitHub account to receive paper recommendations based on both the overall content of your repositories and your recent commits.
                    Stay up-to-date and never miss out on crucial information for your work.
                  </p>
                </div>
                <div>
                  <Button asChild variant="outline" size="sm" className="greek-button">
                    <Link href="/github">Consult Oracle of GitHub</Link>
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
