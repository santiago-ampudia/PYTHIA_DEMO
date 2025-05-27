"use client"

import { usePathname, useRouter } from "next/navigation"
import Link from "next/link"
import { User, MessageSquare, Search, Github, Bell, Home, LogOut, Scroll } from "lucide-react"
import { signOut, useSession } from "next-auth/react"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { ThemeToggle } from "@/components/theme-toggle"

export function AppSidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const { data: session } = useSession()

  const isActive = (path: string) => {
    return pathname === path
  }

  const handleLogout = async () => {
    await signOut({ redirect: true, callbackUrl: "/login" })
    // No need for manual redirect or toast as we're using redirect: true
  }

  return (
    <Sidebar>
      <SidebarHeader className="flex flex-col gap-4 px-4 py-3">
        <div className="flex items-center gap-2 font-semibold">
          <Scroll className="h-6 w-6" />
          <span className="text-xl">PYTHIA</span>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild isActive={isActive("/dashboard")}>
              <Link href="/dashboard">
                <Home />
                <span>Dashboard</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton asChild isActive={isActive("/personal")}>
              <Link href="/personal">
                <User />
                <span>Personal</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton asChild isActive={isActive("/smart-answer")}>
              <Link href="/smart-answer">
                <MessageSquare />
                <span>Ask the Oracle</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton asChild isActive={isActive("/smart-search")}>
              <Link href="/smart-search">
                <Search />
                <span>Pythia's Recommendations</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton asChild isActive={isActive("/github")}>
              <Link href="/github">
                <Github />
                <span>Oracle of GitHub</span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarContent>
      <SidebarFooter className="p-4">
        <div className="flex items-center justify-between">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="h-8 w-8 rounded-full p-0">
                <Avatar className="h-8 w-8">
                  <AvatarImage
                    src={session?.user?.image || "/abstract-geometric-shapes.png"}
                    alt={session?.user?.name || "User"}
                  />
                  <AvatarFallback>{session?.user?.name?.substring(0, 2) || "US"}</AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>{session?.user?.name || "My Account"}</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem asChild>
                <Link href="/personal">Profile</Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href="/personal/settings">Settings</Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={handleLogout} className="text-destructive">
                <LogOut className="h-4 w-4 mr-2" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <Bell className="h-5 w-5" />
              <span className="sr-only">Notifications</span>
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  )
}
