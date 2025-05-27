"use client"

import { useState } from "react"
import { ChevronLeft, LogOut } from "lucide-react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { toast } from "@/components/ui/use-toast"
import { signOut } from "next-auth/react"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"

export default function SettingsPage() {
  const [loading, setLoading] = useState(false)

  const handleSave = () => {
    setLoading(true)
    setTimeout(() => {
      setLoading(false)
      toast({
        title: "Settings saved",
        description: "Your settings have been saved successfully.",
      })
    }, 1000)
  }

  const handleLogout = async () => {
    await signOut({ redirect: true, callbackUrl: "/login" })
  }

  return (
    <div className="container max-w-4xl mx-auto p-4">
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm">
              <Link href="/personal">
                <ChevronLeft className="h-4 w-4" />
                Back
              </Link>
            </Button>
            <h1 className="text-2xl font-bold">Settings</h1>
          </div>

          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" className="gap-2">
                <LogOut className="h-4 w-4" />
                Logout
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Are you sure you want to logout?</AlertDialogTitle>
                <AlertDialogDescription>You will need to sign in again to access your account.</AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={handleLogout}>Logout</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>

        <Tabs defaultValue="profile" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="profile">Profile</TabsTrigger>
            <TabsTrigger value="account">Account</TabsTrigger>
            <TabsTrigger value="notifications">Notifications</TabsTrigger>
            <TabsTrigger value="preferences">Preferences</TabsTrigger>
          </TabsList>

          <TabsContent value="profile" className="mt-4 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Profile Information</CardTitle>
                <CardDescription>Update your profile information visible to other users</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Full Name</Label>
                    <Input id="name" defaultValue="John Researcher" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="username">Username</Label>
                    <Input id="username" defaultValue="johnresearcher" />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="bio">Bio</Label>
                  <Textarea id="bio" defaultValue="AI & Machine Learning Enthusiast" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="institution">Institution</Label>
                  <Input id="institution" defaultValue="University of Research" />
                </div>
              </CardContent>
              <CardFooter>
                <Button onClick={handleSave} disabled={loading}>
                  {loading ? "Saving..." : "Save Changes"}
                </Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Research Interests</CardTitle>
                <CardDescription>Select your research interests to get better recommendations</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center space-x-2">
                    <Switch id="interest-ai" defaultChecked />
                    <Label htmlFor="interest-ai">Artificial Intelligence</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch id="interest-ml" defaultChecked />
                    <Label htmlFor="interest-ml">Machine Learning</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch id="interest-nlp" defaultChecked />
                    <Label htmlFor="interest-nlp">Natural Language Processing</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch id="interest-cv" />
                    <Label htmlFor="interest-cv">Computer Vision</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch id="interest-robotics" />
                    <Label htmlFor="interest-robotics">Robotics</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch id="interest-rl" />
                    <Label htmlFor="interest-rl">Reinforcement Learning</Label>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button onClick={handleSave} disabled={loading}>
                  {loading ? "Saving..." : "Save Changes"}
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="account" className="mt-4 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Account Settings</CardTitle>
                <CardDescription>Update your account settings and password</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" type="email" defaultValue="john.researcher@example.com" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="current-password">Current Password</Label>
                  <Input id="current-password" type="password" />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="new-password">New Password</Label>
                    <Input id="new-password" type="password" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confirm-password">Confirm New Password</Label>
                    <Input id="confirm-password" type="password" />
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button onClick={handleSave} disabled={loading}>
                  {loading ? "Saving..." : "Update Password"}
                </Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Danger Zone</CardTitle>
                <CardDescription>Irreversible account actions</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <h3 className="font-medium">Delete Account</h3>
                  <p className="text-sm text-muted-foreground">
                    Once you delete your account, there is no going back. Please be certain.
                  </p>
                </div>
              </CardContent>
              <CardFooter>
                <Button variant="destructive">Delete Account</Button>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="notifications" className="mt-4 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Notification Settings</CardTitle>
                <CardDescription>Manage how you receive notifications</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-4">
                  <h3 className="font-medium">Email Notifications</h3>
                  <div className="grid grid-cols-1 gap-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="email-papers">New paper recommendations</Label>
                      <Switch id="email-papers" defaultChecked />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label htmlFor="email-answers">Answers to your questions</Label>
                      <Switch id="email-answers" defaultChecked />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label htmlFor="email-upvotes">Upvotes on your answers</Label>
                      <Switch id="email-upvotes" />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label htmlFor="email-digest">Weekly digest</Label>
                      <Switch id="email-digest" defaultChecked />
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="font-medium">In-App Notifications</h3>
                  <div className="grid grid-cols-1 gap-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="app-papers">New paper recommendations</Label>
                      <Switch id="app-papers" defaultChecked />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label htmlFor="app-answers">Answers to your questions</Label>
                      <Switch id="app-answers" defaultChecked />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label htmlFor="app-upvotes">Upvotes on your answers</Label>
                      <Switch id="app-upvotes" defaultChecked />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label htmlFor="app-github">GitHub activity updates</Label>
                      <Switch id="app-github" defaultChecked />
                    </div>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button onClick={handleSave} disabled={loading}>
                  {loading ? "Saving..." : "Save Changes"}
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>

          <TabsContent value="preferences" className="mt-4 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Display Preferences</CardTitle>
                <CardDescription>Customize how content is displayed</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="papers-per-page">Papers per page</Label>
                  <Select defaultValue="10">
                    <SelectTrigger id="papers-per-page">
                      <SelectValue placeholder="Select number of papers" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="5">5</SelectItem>
                      <SelectItem value="10">10</SelectItem>
                      <SelectItem value="20">20</SelectItem>
                      <SelectItem value="50">50</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="default-sort">Default sort order</Label>
                  <Select defaultValue="relevance">
                    <SelectTrigger id="default-sort">
                      <SelectValue placeholder="Select sort order" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="relevance">Relevance</SelectItem>
                      <SelectItem value="date-desc">Newest first</SelectItem>
                      <SelectItem value="date-asc">Oldest first</SelectItem>
                      <SelectItem value="citations">Most cited</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-4">
                  <h3 className="font-medium">Content Display</h3>
                  <div className="grid grid-cols-1 gap-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="show-abstracts">Show paper abstracts in lists</Label>
                      <Switch id="show-abstracts" defaultChecked />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label htmlFor="show-citations">Show citation counts</Label>
                      <Switch id="show-citations" defaultChecked />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label htmlFor="compact-view">Use compact view</Label>
                      <Switch id="compact-view" />
                    </div>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button onClick={handleSave} disabled={loading}>
                  {loading ? "Saving..." : "Save Changes"}
                </Button>
              </CardFooter>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
