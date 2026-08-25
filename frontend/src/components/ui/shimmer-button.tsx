"use client";

import * as React from "react";

import { motion, type HTMLMotionProps } from "motion/react";

import { cn } from "@/lib/utils";

interface ShimmerButtonProps extends HTMLMotionProps<"button"> {
  children: React.ReactNode;
  shimmer?: boolean;
}

function ShimmerButton({ children, className, shimmer = true, ...props }: ShimmerButtonProps) {
  return (
    <motion.button
      className={cn(
        "relative inline-flex overflow-hidden rounded-lg",
        shimmer ? "shimmer-button-sweep" : "bg-primary",
      )}
      whileTap={{
        scale: 0.95,
      }}
      whileHover={{
        scale: 1.05,
      }}
      {...props}
    >
      <span
        className={cn(
          "bg-white m-0.5 rounded-full px-4 py-1 text-sm font-medium text-black backdrop-blur-sm",
          className,
        )}
      >
        {children}
      </span>
    </motion.button>
  );
}

export { ShimmerButton, type ShimmerButtonProps };
