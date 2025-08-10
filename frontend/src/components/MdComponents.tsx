import {cn} from "@/utils.ts";
import {Badge} from "@/components/ui/badge.tsx";
import React, {ReactNode} from "react";
import {Loader2} from "lucide-react";

type MdComponentProps = {
  className?: string;
  children?: ReactNode;
  [key: string]: any;
};

const processChildren = (nodes: React.ReactNode): React.ReactNode => {
    if (!nodes) {
        return null;
    }
    return React.Children.map(nodes, (child) => {
        if (typeof child === "string") {
            if (child.includes("[LOADING_SPINNER]")) {
                return child.split(/(\[LOADING_SPINNER\])/).map((part, index) => {
                    if (part === "[LOADING_SPINNER]") {
                        return <Loader2 key={index} className="h-4 w-4 animate-spin inline-block ml-2" />;
                    }
                    return part;
                });
            }
        }
        if (React.isValidElement(child) && child.props.children) {
            return React.cloneElement(child, {
                ...child.props,
                children: processChildren(child.props.children),
            });
        }
        return child;
    });
}


export const mdComponents = {
  // It ensures the 'src' attribute, with its base64 data, is always passed through.
  img: ({ node, ...props }: MdComponentProps) => {
    // This custom renderer directly accesses the `src` from the parsed markdown 'node'
    // This is the most robust way to ensure the base64 data is never stripped out.
    const imageSource = node?.properties?.src || "";
    const altText = node?.properties?.alt || "";

    return (
      <img
        {...props}
        src={imageSource}
        alt={altText}
        style={{ backgroundBlendMode: "multiply" }} // This is the key change
        className={cn(
          "inline-block h-6 w-6 ml-2 bg-transparent",
          props.className
        )}
      />
    );
  },
  h1: ({ className, children, ...props }: MdComponentProps) => (
    <h1 className={cn("text-2xl font-bold mt-4 mb-2", className)} {...props}>
      {children}
    </h1>
  ),
  h2: ({ className, children, ...props }: MdComponentProps) => (
    <h2 className={cn("text-xl font-bold mt-3 mb-2", className)} {...props}>
      {children}
    </h2>
  ),
  h3: ({ className, children, ...props }: MdComponentProps) => (
    <h3 className={cn("text-lg font-bold mt-3 mb-1", className)} {...props}>
      {children}
    </h3>
  ),
  p: ({ className, children, ...props }: MdComponentProps) => (
    <p className={cn("mb-3 leading-7", className)} {...props}>
      {processChildren(children)}
    </p>
  ),
  a: ({ className, children, href, ...props }: MdComponentProps) => (
    <Badge className="text-xs mx-0.5">
      <a
        className={cn("text-blue-400 hover:text-blue-300 text-xs", className)}
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        {...props}
      >
        {children}
      </a>
    </Badge>
  ),
  ul: ({ className, children, ...props }: MdComponentProps) => (
    <ul className={cn("list-disc pl-6 mb-3", className)} {...props}>
      {children}
    </ul>
  ),
  ol: ({ className, children, ...props }: MdComponentProps) => (
    <ol className={cn("list-decimal pl-6 mb-3", className)} {...props}>
      {children}
    </ol>
  ),
  li: ({ className, children, ...props }: MdComponentProps) => (
    <li className={cn("mb-1 flex items-center", className)} {...props}>
      {processChildren(children)}
    </li>
  ),
  blockquote: ({ className, children, ...props }: MdComponentProps) => (
    <blockquote
      className={cn(
        "border-l-4 border-neutral-600 pl-4 italic my-3 text-sm",
        className
      )}
      {...props}
    >
      {children}
    </blockquote>
  ),
  code: ({ className, children, ...props }: MdComponentProps) => (
    <code
      className={cn(
        "bg-neutral-900 rounded px-1 py-0.5 font-mono text-xs",
        className
      )}
      {...props}
    >
      {children}
    </code>
  ),
  pre: ({ className, children, ...props }: MdComponentProps) => (
    <pre
      className={cn(
        "bg-neutral-900 p-3 rounded-lg overflow-x-auto font-mono text-xs my-3",
        className
      )}
      {...props}
    >
      {children}
    </pre>
  ),
  hr: ({ className, ...props }: MdComponentProps) => (
    <hr className={cn("border-neutral-600 my-4", className)} {...props} />
  ),
  table: ({ className, children, ...props }: MdComponentProps) => (
    <div className="my-3 overflow-x-auto">
      <table className={cn("border-collapse w-full", className)} {...props}>
        {children}
      </table>
    </div>
  ),
  video: ({ className, src, ...props }: MdComponentProps) => (
    <video className={cn("w-full", className)} controls {...props}>
      <source src={src} type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  ),
  th: ({ className, children, ...props }: MdComponentProps) => (
    <th
      className={cn(
        "border border-neutral-600 px-3 py-2 text-left font-bold",
        className
      )}
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ className, children, ...props }: MdComponentProps) => (
    <td
      className={cn("border border-neutral-600 px-3 py-2", className)}
      {...props}
    >
      {children}
    </td>
  ),
};