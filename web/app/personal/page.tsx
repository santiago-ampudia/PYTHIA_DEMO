import Link from "next/link"
import { Settings, Star, Bookmark } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { PaperCard } from "@/components/paper-card"
import { getSavedPapers } from "@/lib/actions/paper"
import { getServerSession } from "next-auth/next"
import { authOptions } from "@/app/api/auth/[...nextauth]/route"
import prisma from "@/lib/db"

export default async function PersonalPage() {
  const session = await getServerSession(authOptions)

  if (!session?.user?.id) {
    return (
      <div className="container max-w-7xl mx-auto p-4">
        <div className="flex flex-col items-center justify-center h-[70vh] gap-4">
          <h1 className="text-2xl font-bold">Please sign in to view your personal dashboard</h1>
          <Button asChild>
            <Link href="/login">Sign In</Link>
          </Button>
        </div>
      </div>
    )
  }

  // Get user data
  const user = await prisma.user.findUnique({
    where: {
      id: session.user.id,
    },
    include: {
      interests: true,
    },
  })

  // Get saved papers
  const savedPapersResult = await getSavedPapers()
  const savedPapers = savedPapersResult.success ? savedPapersResult.data : []

  // Get recommended papers (in a real app, this would be based on user interests)
  const recommendedPapers = await prisma.paper.findMany({
    take: 4,
    orderBy: {
      citations: "desc",
    },
  })

  return (
    <div className="container max-w-7xl mx-auto p-4">
      <div className="flex flex-col gap-6">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div className="flex items-center gap-4">
            <Avatar className="h-16 w-16">
              <AvatarImage src={user?.image || "/abstract-geometric-shapes.png"} alt={user?.name || "User"} />
              <AvatarFallback>{user?.name?.substring(0, 2) || "US"}</AvatarFallback>
            </Avatar>
            <div>
              <h1 className="text-2xl font-bold">{user?.name || "Researcher"}</h1>
              <p className="text-muted-foreground">{user?.bio || "Research Enthusiast"}</p>
            </div>
          </div>
          <Button asChild variant="outline" size="sm">
            <Link href="/personal/settings">
              <Settings className="mr-2 h-4 w-4" />
              Settings
            </Link>
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Saved Papers</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{savedPapers.length}</div>
            </CardContent>
            <CardFooter>
              <Link href="/personal/saved" className="text-sm text-primary flex items-center">
                <Bookmark className="mr-1 h-4 w-4" />
                View all saved
              </Link>
            </CardFooter>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Research Interests</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{user?.interests.length || 0}</div>
            </CardContent>
            <CardFooter>
              <Link href="/personal/settings" className="text-sm text-primary flex items-center">
                <Star className="mr-1 h-4 w-4" />
                Manage interests
              </Link>
            </CardFooter>
          </Card>
        </div>

        <Tabs defaultValue="recommendations" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
            <TabsTrigger value="saved">Saved Papers</TabsTrigger>
          </TabsList>
          <TabsContent value="recommendations" className="mt-4">
            <h2 className="text-xl font-semibold mb-4">Recommended for you</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {recommendedPapers.map((paper) => (
                <PaperCard
                  key={paper.id}
                  id={paper.id}
                  title={paper.title}
                  authors={paper.authors}
                  journal={paper.journal || ""}
                  abstract={paper.abstract || ""}
                  citations={paper.citations || 0}
                  year={paper.year}
                  field={paper.field || ""}
                  isSaved={savedPapers.some((sp: any) => sp.paperId === paper.id)}
                />
              ))}
            </div>
          </TabsContent>
          <TabsContent value="saved" className="mt-4">
            <h2 className="text-xl font-semibold mb-4">Your saved papers</h2>
            {savedPapers.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {savedPapers.map((savedPaper: any) => (
                  <PaperCard
                    key={savedPaper.paper.id}
                    id={savedPaper.paper.id}
                    title={savedPaper.paper.title}
                    authors={savedPaper.paper.authors}
                    journal={savedPaper.paper.journal || ""}
                    abstract={savedPaper.paper.abstract || ""}
                    citations={savedPaper.paper.citations || 0}
                    year={savedPaper.paper.year}
                    field={savedPaper.paper.field || ""}
                    isSaved={true}
                    notes={savedPaper.notes}
                  />
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-muted-foreground">
                  You haven't saved any papers yet. Start by searching for papers or asking questions in Smart Search or
                  Smart Answer.
                </p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
