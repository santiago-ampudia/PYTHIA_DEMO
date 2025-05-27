"use client"

import { useState, useEffect } from "react"
import { Github, ArrowRight, BookOpen, Star, GitFork, Code, Calendar, Clock, BookText, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { PaperCard } from "@/components/paper-card"
import { CategorizedTweetRecommendations } from "@/components/categorized-tweet-recommendations"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { toast } from "@/components/ui/use-toast"
import { signIn, useSession } from "next-auth/react"
import { Skeleton } from "@/components/ui/skeleton"

// Types for GitHub API responses
interface GitHubRepo {
  id: number
  name: string
  description: string
  html_url: string
  stargazers_count: number
  forks_count: number
  language: string
  updated_at: string
  owner: {
    login: string
    avatar_url: string
  }
}

interface GitHubCommit {
  sha: string
  commit: {
    message: string
    author: {
      name: string
      date: string
    }
  }
  html_url: string
  repository: {
    name: string
  }
}

interface GitHubUser {
  login: string
  avatar_url: string
  name: string
  bio: string
  html_url: string
}

// Interface for paper recommendation data
interface PaperRecommendation {
  tweet_text: string
  score: number
  paper_ids: string[]
  batch_id: string
  component_scores: {
    tweet_relevance_score: number
    avg_normalized_weight: number
    lambda_weight: number
  }
}

interface RecommendationData {
  query: string
  timestamp: string
  recommendations: PaperRecommendation[]
}

export default function GitHubPage() {
  const { data: session, status } = useSession()
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isCheckingConnection, setIsCheckingConnection] = useState(true)
  const [githubUser, setGithubUser] = useState<GitHubUser | null>(null)
  const [repos, setRepos] = useState<GitHubRepo[]>([])
  const [commits, setCommits] = useState<GitHubCommit[]>([])
  const [activeTab, setActiveTab] = useState("repositories")
  const [recommendations, setRecommendations] = useState<PaperRecommendation[]>([])
  const [categorizedRecommendations, setCategorizedRecommendations] = useState<any>(null)
  const [recommendationQuery, setRecommendationQuery] = useState<string>("") 
  const [recommendationTimestamp, setRecommendationTimestamp] = useState<string>("") 
  const [isLoadingRecommendations, setIsLoadingRecommendations] = useState(false)
  const [isCachedRecommendation, setIsCachedRecommendation] = useState(false)
  const [selectedRepo, setSelectedRepo] = useState<string>("")   
  // State for recent work recommendations
  const [recentWorkRecommendations, setRecentWorkRecommendations] = useState<PaperRecommendation[]>([])
  const [categorizedRecentWorkRecommendations, setCategorizedRecentWorkRecommendations] = useState<any>(null)
  const [recentWorkQuery, setRecentWorkQuery] = useState<string>("") 
  const [recentWorkTimestamp, setRecentWorkTimestamp] = useState<string>("") 
  const [isLoadingRecentWorkRecommendations, setIsLoadingRecentWorkRecommendations] = useState(false)

  // Check if the URL has a code parameter (GitHub OAuth callback)
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search)
    const code = urlParams.get("code")

    if (code) {
      // Remove the code from the URL to prevent issues on refresh
      window.history.replaceState({}, document.title, window.location.pathname)

      // If we have a code, we're coming back from GitHub OAuth
      // The actual token exchange happens on the server via NextAuth
      toast({
        title: "GitHub Connected",
        description: "Successfully connected to GitHub. Loading your data...",
      })

      // We'll check connection status which will load the data
      checkGitHubConnection()
    } else {
      // Normal page load, check connection
      checkGitHubConnection()
    }
  }, [])

  // Check connection status whenever session changes
  useEffect(() => {
    if (status !== "loading") {
      checkGitHubConnection()
    }
  }, [status])

  const checkGitHubConnection = async () => {
    if (status === "loading") return

    setIsCheckingConnection(true)
    try {
      const response = await fetch("/api/github/status")
      const data = await response.json()

      if (data.connected) {
        setIsConnected(true)
        // Load GitHub data
        loadGitHubData()
      } else {
        setIsConnected(false)
      }
    } catch (error) {
      console.error("Error checking GitHub connection:", error)
      setIsConnected(false)
    } finally {
      setIsCheckingConnection(false)
    }
  }

  const loadGitHubData = async () => {
    setIsLoading(true)
    try {
      // Load GitHub user profile
      const userResponse = await fetch("/api/github/user")
      const userData = await userResponse.json()
      setGithubUser(userData)

      // Load repositories
      const reposResponse = await fetch("/api/github/repos")
      const reposData = await reposResponse.json()
      setRepos(reposData)

      // Load recent commits
      const commitsResponse = await fetch("/api/github/commits")
      const commitsData = await commitsResponse.json()
      setCommits(commitsData)

      // Always default to repositories tab
      setActiveTab("repositories")
    } catch (error) {
      console.error("Error loading GitHub data:", error)
      toast({
        title: "Error",
        description: "Failed to load GitHub data. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleConnect = async () => {
    setIsLoading(true)
    try {
      // Use NextAuth to sign in with GitHub
      // This will redirect to GitHub and then back to this page
      await signIn("github", {
        callbackUrl: window.location.href, // Return to this page after auth
        redirect: true,
      })
    } catch (error) {
      console.error("Error connecting to GitHub:", error)
      toast({
        title: "Error",
        description: "Failed to connect to GitHub. Please try again.",
        variant: "destructive",
      })
      setIsLoading(false)
    }
  }

  const handleDisconnect = async () => {
    setIsLoading(true)
    try {
      const response = await fetch("/api/github/disconnect", {
        method: "POST",
      })

      if (response.ok) {
        setIsConnected(false)
        setGithubUser(null)
        setRepos([])
        setCommits([])
        toast({
          title: "Success",
          description: "Your GitHub account has been disconnected.",
        })
      } else {
        throw new Error("Failed to disconnect GitHub account")
      }
    } catch (error) {
      console.error("Error disconnecting from GitHub:", error)
      toast({
        title: "Error",
        description: "Failed to disconnect from GitHub. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  // Format in-text citations as numbered hyperlinks with React elements
  const formatCitations = (text: string, paperIds: string[]) => {
    if (!text || !paperIds || paperIds.length === 0) return text;
    
    // Create a citation map for the paper IDs
    const citationMap = new Map<string, number>();
    paperIds.forEach((id, index) => {
      citationMap.set(id, index + 1);
    });
    
    // Regular expression to match arXiv IDs in square brackets
    const regex = /\[((?:arXiv:)?[a-zA-Z0-9\.-]+\/?\d*)\]/g;
    
    // Split the text into parts with React elements for citations
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match;
    
    while ((match = regex.exec(text)) !== null) {
      // Add text before the match
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      
      const arxivId = match[1];
      const citationNumber = citationMap.get(arxivId) || paperIds.indexOf(arxivId) + 1;
      
      // Only add a link if we have a valid citation number
      if (citationNumber > 0) {
        // Create a link element for the citation
        parts.push(
          <a 
            key={`citation-${match.index}`}
            href={`https://arxiv.org/abs/${arxivId}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline font-medium"
          >
            [{citationNumber}]
          </a>
        );
      } else {
        // If no valid citation, just add the original text
        parts.push(match[0]);
      }
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add any remaining text after the last match
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }
    
    return parts.length > 0 ? <>{parts}</> : text;
  };
  
  // Handle getting literature recommendations for a repository
  const handleGetRecommendations = async (repoName: string) => {
    setIsLoadingRecommendations(true)
    setActiveTab("recommendations")
    
    // Reset recent work recommendations when a new repository is selected
    setRecentWorkRecommendations([])
    setRecentWorkQuery("")
    setRecentWorkTimestamp("")
    
    // Set the selected repository
    setSelectedRepo(repoName)
    
    try {
      // Call the backend to trigger the search algorithm
      // Always force a refresh to generate new recommendations
      const response = await fetch("/api/github/recommendations", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ repoName, forceRefresh: true }),
      })
      
      if (!response.ok) {
        throw new Error("Failed to get recommendations")
      }
      
      const data = await response.json()
      
      // Clear any existing recent work recommendations when getting new general recommendations
      setRecentWorkRecommendations([]);
      setCategorizedRecentWorkRecommendations(null);
      
      // Check if the response contains categorized recommendations
      if (data.categories && Array.isArray(data.categories)) {
        // New format with categories
        setCategorizedRecommendations(data);
        setRecommendations([]);
      } else if (data.recommendations && Array.isArray(data.recommendations)) {
        // Old format with recommendations array
        setCategorizedRecommendations(null);
        
        if (data.recommendations.length > 0 && typeof data.recommendations[0] === 'string') {
          // Convert string recommendations to the expected object format
          const formattedRecommendations = data.recommendations.map((text: string, index: number) => ({
            tweet_text: text,
            score: null,
            paper_ids: [],
            batch_id: `batch-${index}`,
            component_scores: null
          }));
          setRecommendations(formattedRecommendations);
        } else {
          // Already in the expected format
          setRecommendations(data.recommendations);
        }
      } else {
        // No valid data format
        setRecommendations([]);
        setCategorizedRecommendations(null);
      }
      
      setRecommendationQuery(data.query || "")
      setRecommendationTimestamp(data.timestamp || "")
      setIsCachedRecommendation(data.fromCache || false)
      
      toast({
        title: data.fromCache ? "Cached Recommendations" : "Recommendations Ready",
        description: `${data.fromCache ? "Retrieved" : "Generated"} ${data.recommendations?.length || 0} paper recommendations for ${repoName}`,
        variant: "default"
      })
    } catch (error) {
      console.error("Error getting recommendations:", error)
      toast({
        title: "Error",
        description: "Failed to get paper recommendations. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoadingRecommendations(false)
    }
  }
  
  // Handle getting literature recommendations based on recent work
  const handleGetRecentWorkRecommendations = async () => {
    // Keep existing recommendations and add recent work recommendations below
    // Instead of replacing them
    setIsLoadingRecentWorkRecommendations(true)
    
    try {
      // Call the backend to trigger the search algorithm for recent work
      const response = await fetch("/api/github/recommendations-recent-work", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ repoName: selectedRepo }),
      })
      
      if (!response.ok) {
        throw new Error("Failed to get recent work recommendations")
      }
      
      const data = await response.json()
      
      // Check if the response contains categorized recommendations
      if (data.categories && Array.isArray(data.categories)) {
        // New format with categories for recent work
        setCategorizedRecentWorkRecommendations(data);
        setRecentWorkRecommendations([]);
      } else if (data.recommendations && Array.isArray(data.recommendations)) {
        // Old format with recommendations array
        setCategorizedRecentWorkRecommendations(null);
        
        if (data.recommendations.length > 0 && typeof data.recommendations[0] === 'string') {
          // Convert string recommendations to the expected object format
          const formattedRecommendations = data.recommendations.map((text: string, index: number) => ({
            tweet_text: text,
            score: null,
            paper_ids: [],
            batch_id: `recent-work-batch-${index}`,
            component_scores: null
          }));
          setRecentWorkRecommendations(formattedRecommendations);
        } else {
          // Already in the expected format
          setRecentWorkRecommendations(data.recommendations);
        }
      } else {
        // No valid data format
        setRecentWorkRecommendations([]);
        setCategorizedRecentWorkRecommendations(null);
      }
      
      setRecentWorkQuery(data.query || "")
      setRecentWorkTimestamp(data.timestamp || "")
      
      toast({
        title: "Recent Work Recommendations Ready",
        description: `Generated ${data.categories ? 
          data.categories.reduce((total: number, cat: any) => total + (cat.tweets?.length || 0), 0) : 
          (data.recommendations?.length || 0)} paper recommendations based on recent work in ${selectedRepo}`,
        variant: "default"
      })
    } catch (error) {
      console.error("Error getting recent work recommendations:", error)
      toast({
        title: "Error",
        description: "Failed to get recent work recommendations. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoadingRecentWorkRecommendations(false)
    }
  }
  
  // Format date to relative time (e.g., "2 days ago")
  const formatRelativeTime = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000)

    if (diffInSeconds < 60) return `${diffInSeconds} seconds ago`
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`
    if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`
    if (diffInSeconds < 2592000) return `${Math.floor(diffInSeconds / 604800)} weeks ago`
    if (diffInSeconds < 31536000) return `${Math.floor(diffInSeconds / 2592000)} months ago`
    return `${Math.floor(diffInSeconds / 31536000)} years ago`
  }

  // If we're still checking the initial connection status
  if (isCheckingConnection) {
    return (
      <div className="container max-w-4xl mx-auto p-4">
        <div className="flex flex-col gap-6">
          <div>
            <h1 className="text-2xl font-bold mb-2">GitHub Integration</h1>
            <p className="text-muted-foreground">
              Connect your GitHub account to get paper recommendations based on your repositories
            </p>
          </div>
          <div className="flex flex-col items-center justify-center py-12 space-y-4">
            <Skeleton className="h-12 w-12 rounded-full" />
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-4 w-64" />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="container max-w-4xl mx-auto p-4">
      <div className="flex flex-col gap-6">
        <div>
          <h1 className="text-2xl font-bold mb-2">GitHub Integration</h1>
          <p className="text-muted-foreground">
            Connect your GitHub account to get paper recommendations based on your repositories
          </p>
        </div>

        {!isConnected ? (
          <Card className="border-dashed">
            <CardHeader>
              <CardTitle>Connect your GitHub account</CardTitle>
              <CardDescription>
                Link your GitHub account to get personalized paper recommendations based on your repositories and coding
                activity
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col items-center justify-center py-6">
              <Github className="h-16 w-16 mb-4 text-muted-foreground" />
              <p className="text-center max-w-md mb-4">
                By connecting your GitHub account, we'll analyze your repositories and provide research papers that are
                relevant to your work.
              </p>
            </CardContent>
            <CardFooter className="flex justify-center">
              <Button onClick={handleConnect} disabled={isLoading} className="gap-2">
                <Github className="h-4 w-4" />
                {isLoading ? "Connecting..." : "Connect GitHub Account"}
              </Button>
            </CardFooter>
          </Card>
        ) : (
          <>
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div className="flex items-center gap-4">
                <Avatar className="h-12 w-12">
                  <AvatarImage src={githubUser?.avatar_url || "/github-user-profile.png"} alt="GitHub User" />
                  <AvatarFallback>GH</AvatarFallback>
                </Avatar>
                <div>
                  <h2 className="text-xl font-semibold">{githubUser?.name || "GitHub User"}</h2>
                  <p className="text-muted-foreground">
                    <a
                      href={githubUser?.html_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:underline"
                    >
                      github.com/{githubUser?.login}
                    </a>
                  </p>
                </div>
              </div>
              <Button onClick={handleDisconnect} variant="outline" disabled={isLoading} className="gap-2">
                {isLoading ? "Disconnecting..." : "Disconnect GitHub Account"}
              </Button>
            </div>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="repositories">Your Repositories</TabsTrigger>
                <TabsTrigger value="commits">Recent Commits</TabsTrigger>
                <TabsTrigger value="recommendations">Paper Recommendations</TabsTrigger>
              </TabsList>

              <TabsContent value="repositories" className="mt-4 space-y-4">
                <h2 className="text-xl font-semibold mb-4">Your GitHub Repositories</h2>
                {isLoading ? (
                  <div className="space-y-4">
                    {[1, 2, 3].map((i) => (
                      <Card key={i}>
                        <CardHeader className="pb-2">
                          <Skeleton className="h-6 w-48 mb-2" />
                          <Skeleton className="h-4 w-full" />
                        </CardHeader>
                        <CardFooter className="pt-2 flex justify-between">
                          <div className="flex items-center gap-4">
                            <Skeleton className="h-4 w-16" />
                            <Skeleton className="h-4 w-16" />
                          </div>
                          <Skeleton className="h-8 w-24" />
                        </CardFooter>
                      </Card>
                    ))}
                  </div>
                ) : repos.length > 0 ? (
                  repos.map((repo) => (
                    <Card key={repo.id}>
                      <CardHeader className="pb-2">
                        <div className="flex justify-between items-start">
                          <div>
                            <CardTitle className="text-lg font-semibold flex items-center gap-2">
                              <BookOpen className="h-4 w-4" />
                              {repo.name}
                            </CardTitle>
                            <CardDescription>{repo.description || "No description available"}</CardDescription>
                          </div>
                          {repo.language && <Badge variant="outline">{repo.language}</Badge>}
                        </div>
                      </CardHeader>
                      <CardFooter className="pt-2 flex justify-between">
                        <div className="flex items-center gap-4 text-sm text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Star className="h-4 w-4" />
                            {repo.stargazers_count}
                          </span>
                          <span className="flex items-center gap-1">
                            <GitFork className="h-4 w-4" />
                            {repo.forks_count}
                          </span>
                          <span>Updated {formatRelativeTime(repo.updated_at)}</span>
                        </div>
                        <div className="flex flex-col gap-2">
                          <Button variant="ghost" size="sm" asChild>
                            <a
                              href={repo.html_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center gap-1"
                            >
                              View on GitHub
                              <ArrowRight className="h-4 w-4" />
                            </a>
                          </Button>
                          <Button 
                            variant="outline" 
                            size="sm" 
                            className="flex items-center gap-1"
                            onClick={() => handleGetRecommendations(repo.name)}
                            disabled={isLoadingRecommendations}
                          >
                            <BookText className="h-4 w-4" />
                            See Literature Recommendation
                          </Button>
                        </div>
                      </CardFooter>
                    </Card>
                  ))
                ) : (
                  <div className="text-center py-8">
                    <p className="text-muted-foreground">No repositories found.</p>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="commits" className="mt-4 space-y-4">
                <h2 className="text-xl font-semibold mb-4">Your Recent Commits</h2>
                {isLoading ? (
                  <div className="space-y-4">
                    {[1, 2, 3].map((i) => (
                      <Card key={i}>
                        <CardHeader className="pb-2">
                          <Skeleton className="h-6 w-48 mb-2" />
                          <Skeleton className="h-4 w-full" />
                        </CardHeader>
                        <CardFooter className="pt-2 flex justify-between">
                          <div className="flex items-center gap-4">
                            <Skeleton className="h-4 w-24" />
                            <Skeleton className="h-4 w-24" />
                          </div>
                          <Skeleton className="h-8 w-24" />
                        </CardFooter>
                      </Card>
                    ))}
                  </div>
                ) : commits.length > 0 ? (
                  commits.map((commit) => (
                    <Card key={commit.sha}>
                      <CardHeader className="pb-2">
                        <div className="flex justify-between items-start">
                          <div>
                            <CardTitle className="text-base font-medium flex items-center gap-2">
                              <Code className="h-4 w-4" />
                              {commit.repository.name}
                            </CardTitle>
                            <CardDescription className="line-clamp-2">{commit.commit.message}</CardDescription>
                          </div>
                        </div>
                      </CardHeader>
                      <CardFooter className="pt-2 flex justify-between">
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Calendar className="h-4 w-4" />
                            {new Date(commit.commit.author.date).toLocaleDateString()}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="h-4 w-4" />
                            {new Date(commit.commit.author.date).toLocaleTimeString()}
                          </span>
                        </div>
                        <Button variant="ghost" size="sm" asChild>
                          <a
                            href={commit.html_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1"
                          >
                            View Commit
                            <ArrowRight className="h-4 w-4" />
                          </a>
                        </Button>
                      </CardFooter>
                    </Card>
                  ))
                ) : (
                  <div className="text-center py-8">
                    <p className="text-muted-foreground">No recent commits found.</p>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="recommendations" className="mt-4 space-y-4">
                <h2 className="text-xl font-semibold mb-4">Recommended Papers Based on Your Repositories</h2>
                {isLoadingRecommendations ? (
                  <div className="space-y-4">
                    {[1, 2, 3].map((i) => (
                      <Card key={i}>
                        <CardHeader className="pb-2">
                          <Skeleton className="h-6 w-48 mb-2" />
                          <Skeleton className="h-4 w-full" />
                        </CardHeader>
                        <CardContent>
                          <Skeleton className="h-24 w-full" />
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                ) : categorizedRecommendations ? (
                  <div className="space-y-4">
                    {/* Main repository recommendations header */}
                    <Card className="bg-muted/50 border-2 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.5)] animate-pulse">
                      <CardHeader>
                        <CardTitle className="text-lg">Recommended literature based on overall understanding of {selectedRepo || "repository"}</CardTitle>
                      </CardHeader>
                      <CardFooter className="flex justify-between items-center">
                        <p className="text-sm text-muted-foreground">Generated on: {new Date(recommendationTimestamp).toLocaleString()}</p>
                        {isCachedRecommendation && (
                          <Badge variant="outline" className="ml-2">
                            <Clock className="h-3 w-3 mr-1" /> Cached
                          </Badge>
                        )}
                      </CardFooter>
                    </Card>
                    
                    {/* Categorized recommendations */}
                    <CategorizedTweetRecommendations categories={categorizedRecommendations.categories} />
                    
                    {/* Button to get recent work recommendations */}
                    {!isLoadingRecentWorkRecommendations && !categorizedRecentWorkRecommendations && recentWorkRecommendations.length === 0 && (
                      <div className="flex justify-center mt-6">
                        <Button 
                          onClick={handleGetRecentWorkRecommendations} 
                          disabled={isLoadingRecentWorkRecommendations}
                          className="flex items-center gap-2"
                        >
                          <BookText className="h-4 w-4" />
                          See recommended literature based on your recent work
                        </Button>
                      </div>
                    )}
                    
                    {/* Loading state for recent work recommendations */}
                    {isLoadingRecentWorkRecommendations && (
                      <div className="space-y-4 mt-6">
                        <Card className="bg-muted/50 border-2 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.5)] animate-pulse">
                          <CardHeader>
                            <CardTitle className="text-lg">Loading recommendations based on recent work...</CardTitle>
                          </CardHeader>
                        </Card>
                        {[1, 2, 3].map((i) => (
                          <Card key={`loading-recent-${i}`}>
                            <CardHeader className="pb-2">
                              <Skeleton className="h-6 w-48 mb-2" />
                              <Skeleton className="h-4 w-full" />
                            </CardHeader>
                            <CardContent>
                              <Skeleton className="h-24 w-full" />
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    )}
                    
                    {/* Recent work recommendations */}
                    {!isLoadingRecentWorkRecommendations && (categorizedRecentWorkRecommendations || recentWorkRecommendations.length > 0) && (
                      <div className="space-y-4 mt-6">
                        <Card className="bg-muted/50 border-2 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.5)] animate-pulse">
                          <CardHeader>
                            <CardTitle className="text-lg">Recommended literature based on recent work</CardTitle>
                          </CardHeader>
                          <CardFooter className="flex justify-between items-center">
                            <p className="text-sm text-muted-foreground">Generated on: {new Date(recentWorkTimestamp).toLocaleString()}</p>
                          </CardFooter>
                        </Card>
                        
                        {categorizedRecentWorkRecommendations ? (
                          /* Categorized recent work recommendations */
                          <CategorizedTweetRecommendations categories={categorizedRecentWorkRecommendations.categories} />
                        ) : (
                          /* Legacy format recent work recommendations */
                          recentWorkRecommendations.map((recommendation, index) => (
                            <Card key={`recent-${index}`}>
                              <CardContent className="pt-4">
                                <p className="mb-3">
                                  {recommendation.paper_ids && recommendation.paper_ids.length > 0 
                                    ? formatCitations(recommendation.tweet_text || '', recommendation.paper_ids)
                                    : (recommendation.tweet_text || 'No recommendation text available')
                                  }
                                </p>
                                {recommendation.paper_ids && recommendation.paper_ids.length > 0 ? (
                                  <div className="flex flex-wrap gap-2">
                                    {recommendation.paper_ids.map((paperId) => (
                                      <a 
                                        key={paperId} 
                                        href={`https://arxiv.org/abs/${paperId}`} 
                                        target="_blank" 
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center gap-1 text-blue-600 hover:underline font-medium"
                                      >
                                        <ExternalLink className="h-3 w-3" />
                                        <span>arXiv ID: {paperId}</span>
                                      </a>
                                    ))}
                                  </div>
                                ) : null}
                              </CardContent>
                            </Card>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                ) : recommendations.length > 0 ? (
                  <div className="space-y-4">
                    {/* Main repository recommendations */}
                    <Card className="bg-muted/50 border-2 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.5)] animate-pulse">
                      <CardHeader>
                        <CardTitle className="text-lg">Recommended literature based on overall understanding of {selectedRepo || "repository"}</CardTitle>
                      </CardHeader>
                      <CardFooter className="flex justify-between items-center">
                        <p className="text-sm text-muted-foreground">Generated on: {new Date(recommendationTimestamp).toLocaleString()}</p>
                        {isCachedRecommendation && (
                          <Badge variant="outline" className="ml-2">
                            <Clock className="h-3 w-3 mr-1" /> Cached
                          </Badge>
                        )}
                      </CardFooter>
                    </Card>
                    
                    {/* Repository recommendations (legacy format) */}
                    {recommendations.map((recommendation, index) => (
                      <Card key={`repo-${index}`}>
                        <CardContent className="pt-4">
                          <p className="mb-3">
                            {recommendation.paper_ids && recommendation.paper_ids.length > 0 
                              ? formatCitations(recommendation.tweet_text || '', recommendation.paper_ids)
                              : (recommendation.tweet_text || 'No recommendation text available')
                            }
                          </p>
                          {recommendation.paper_ids && recommendation.paper_ids.length > 0 ? (
                            <div className="flex flex-wrap gap-2">
                              {recommendation.paper_ids.map((paperId) => (
                                <a 
                                  key={paperId} 
                                  href={`https://arxiv.org/abs/${paperId}`} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1 text-blue-600 hover:underline font-medium"
                                >
                                  <ExternalLink className="h-3 w-3" />
                                  <span>arXiv ID: {paperId}</span>
                                </a>
                              ))}
                            </div>
                          ) : null}
                        </CardContent>
                      </Card>
                    ))}
                    
                    {/* Button to get recent work recommendations */}
                    {!isLoadingRecentWorkRecommendations && !categorizedRecentWorkRecommendations && recentWorkRecommendations.length === 0 && (
                      <div className="flex justify-center mt-6">
                        <Button 
                          onClick={handleGetRecentWorkRecommendations} 
                          disabled={isLoadingRecentWorkRecommendations}
                          className="flex items-center gap-2"
                        >
                          <BookText className="h-4 w-4" />
                          See recommended literature based on your recent work
                        </Button>
                      </div>
                    )}
                    
                    {/* Loading state for recent work recommendations */}
                    {isLoadingRecentWorkRecommendations && (
                      <div className="space-y-4 mt-6">
                        <Card className="bg-muted/50 border-2 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.5)] animate-pulse">
                          <CardHeader>
                            <CardTitle className="text-lg">Loading recommendations based on recent work...</CardTitle>
                          </CardHeader>
                        </Card>
                        {[1, 2, 3].map((i) => (
                          <Card key={`loading-recent-${i}`}>
                            <CardHeader className="pb-2">
                              <Skeleton className="h-6 w-48 mb-2" />
                              <Skeleton className="h-4 w-full" />
                            </CardHeader>
                            <CardContent>
                              <Skeleton className="h-24 w-full" />
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    )}
                    
                    {/* Recent work recommendations */}
                    {!isLoadingRecentWorkRecommendations && (categorizedRecentWorkRecommendations || recentWorkRecommendations.length > 0) && (
                      <div className="space-y-4 mt-6">
                        <Card className="bg-muted/50 border-2 border-blue-500 shadow-[0_0_15px_rgba(59,130,246,0.5)] animate-pulse">
                          <CardHeader>
                            <CardTitle className="text-lg">Recommended literature based on recent work</CardTitle>
                          </CardHeader>
                          <CardFooter className="flex justify-between items-center">
                            <p className="text-sm text-muted-foreground">Generated on: {new Date(recentWorkTimestamp).toLocaleString()}</p>
                          </CardFooter>
                        </Card>
                        
                        {categorizedRecentWorkRecommendations ? (
                          /* Categorized recent work recommendations */
                          <CategorizedTweetRecommendations categories={categorizedRecentWorkRecommendations.categories} />
                        ) : (
                          /* Legacy format recent work recommendations */
                          recentWorkRecommendations.map((recommendation, index) => (
                            <Card key={`recent-${index}`}>
                              <CardContent className="pt-4">
                                <p className="mb-3">
                                  {recommendation.paper_ids && recommendation.paper_ids.length > 0 
                                    ? formatCitations(recommendation.tweet_text || '', recommendation.paper_ids)
                                    : (recommendation.tweet_text || 'No recommendation text available')
                                  }
                                </p>
                                {recommendation.paper_ids && recommendation.paper_ids.length > 0 ? (
                                  <div className="flex flex-wrap gap-2">
                                    {recommendation.paper_ids.map((paperId) => (
                                      <a 
                                        key={paperId} 
                                        href={`https://arxiv.org/abs/${paperId}`} 
                                        target="_blank" 
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center gap-1 text-blue-600 hover:underline font-medium"
                                      >
                                        <ExternalLink className="h-3 w-3" />
                                        <span>arXiv ID: {paperId}</span>
                                      </a>
                                    ))}
                                  </div>
                                ) : null}
                              </CardContent>
                            </Card>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p className="text-muted-foreground mb-4">No recommendations yet. Click "See Literature Recommendation" on any of your repositories to get personalized paper recommendations.</p>
                    <p className="text-sm text-muted-foreground">Recommendations are saved to your account and can be accessed later without regenerating them.</p>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </>
        )}
      </div>
    </div>
  )
}
