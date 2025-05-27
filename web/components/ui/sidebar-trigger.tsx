"use client"

import { Menu } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useSidebar } from "@/components/ui/sidebar"

export function SidebarTrigger() {
  const { onOpen } = useSidebar()

  return (
    <Button variant="ghost" size="icon" onClick={onOpen}>
      <Menu className="h-5 w-5" />
      <span className="sr-only">Open sidebar</span>
    </Button>
  )
}
