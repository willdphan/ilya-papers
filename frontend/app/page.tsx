"use client";

import React, { useState, useEffect } from 'react';
import { useRef } from "react";
import { useChat } from "ai/react";
import va from "@vercel/analytics";
import clsx from "clsx";
import { VercelIcon, GithubIcon, LoadingCircle, SendIcon } from "./icons";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Textarea from "react-textarea-autosize";
import { toast } from "sonner";
import './globals.css';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';


const examples = [
  "In Ilya's list, what is the most cited paper?",
  "Summarize the 'Attention is All You Need' paper.",
  "Provide list of Ilya's recommended research papers.",
];


export default function Chat() {
  const formRef = useRef<HTMLFormElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  type Message = {
    role: "user" | "assistant";
    content: string;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() === "") return;

    setIsLoading(true);
    const userMessage: Message = { role: "user", content: input };
    setMessages([...messages, userMessage]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      const botMessage: Message = { role: "assistant", content: data.response };
      setMessages([...messages, userMessage, botMessage]);
    } catch (error) {
      console.error('Fetch error:', error);
      toast.error("Error fetching data from local server");
    } finally {
      setIsLoading(false);
      setInput("");
    }
  };

  const disabled = isLoading || input.length === 0;

  // Define the variants for the container to control the children's animation
const containerVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.1, // Delay between each child animation
    },
  },
};

// Define the variants for each child to fade in
const childVariants = {
  hidden: { opacity: 0},
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.5 }, // Duration of the fade-in effect
  },
};

const nextSectionRef = useRef<HTMLDivElement>(null);

const scrollToNextSection = () => {
  // Now TypeScript knows that current is an HTMLDivElement or null
  nextSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
};






  return (
    <main >
    {/* <div className="min-h-screen w-screen bg-cover bg-center" style={{ backgroundImage: `url('/ilya.jpeg')` }}>
    <div className="flex h-full w-full items-center justify-center">
      <div className="text-white text-center">
        <h1 className="text-4xl font-bold mb-2">Welcome to Our Site</h1>
        <p className="text-xl">Explore our services and products.</p>
      </div>
    </div>
  </div> */}


<div className='absolute top-8 pl-8 flex items-center justify-between w-full'>
<div className="fixed top-8 left-8 z-50">
  <button className='no-icon' onClick={() => document.getElementById('first-section')?.scrollIntoView({ behavior: 'smooth' })}>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#1871E3" width="30" height="30" className="default-icon">
      <path d="M18 7H22V9H16V3H18V7ZM8 9H2V7H6V3H8V9ZM18 17V21H16V15H22V17H18ZM8 15V21H6V17H2V15H8Z"></path>
    </svg>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#1871E3" width="30" height="30" className="hover-icon hidden">
      <path d="M18 7H22V9H16V3H18V7ZM8 9H2V7H6V3H8V9ZM18 17V21H16V15H22V17H18ZM8 15V21H6V17H2V15H8Z"></path>
    </svg>
  </button>
</div>
 
  <div className="hidden flex-grow sm:flex justify-center items-center space-x-4 font-[300]">
  {/* <Link href="/about"> */}
    <button className="text-[#E5E5E5] hover:text-[#1871E3] mr-8 no-icon text-xl ">
      About
    
    </button>
  {/* </Link> */}
  <a href="mailto:willdphan@gmail.com">
    <button className="text-[#E5E5E5] hover:text-[#1871E3] no-icon text-xl">
      Contact
    </button>
  </a>
</div>


</div>

{/* VIDEO */}
<div id="first-section" className='min-h-screen px-8 pt-24 bg-[#0A0A0A] space-y-10 font-Sans '>
  <div className='flex flex-col items-center'>
  <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          whileHover={{ scale: 1 }}  
          transition={{ duration: 2 }} className="w-full h-[50vh] sm:h-[50vh] md:h-[55vh] lg:h-[55vh] xl:h-[55vh] 2xl:h-[55vh]">
      <video className='object-cover w-full h-full' autoPlay loop muted playsInline>
        <source src="https://pub-33c643825c664d0091b84d7ae37a5150.r2.dev/ilya-display.mp4" type="video/mp4" />
      </video>
    </motion.div>
  </div>


    
  <div className='flex flex-col sm:flex-row w-full '>
  <div className='hidden sm:flex items-start w-[100%] sm:w-[50%] justify-start items-end'>
  <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          whileHover={{ scale: 1 }}  
          transition={{ duration: 2 }}
        >
          <button 
            className="flex items-start text-xl text-[#1871E3] no-icon "
            onClick={scrollToNextSection}  // Add the onClick handler here
          >
            Check it out
            <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" viewBox="0 0 24 24" fill="currentColor" className="ml-2 mt-[2px]">
              <path d="M13.0001 16.1716L18.3641 10.8076L19.7783 12.2218L12.0001 20L4.22192 12.2218L5.63614 10.8076L11.0001 16.1716V4H13.0001V16.1716Z"></path>
            </svg>
          </button>
        </motion.div>
</div  >



<div className='flex sm:w-1/2'>
  <div
    style={{ lineHeight: '1.2' }}
    className='w-full h-full flex justify-start items-end text-4xl sm:text-4xl md:text-4xl lg:text-5xl xl:text-6xl font-Sans text-[#E5E5E5] flex-wrap font-[400]'
  >
    {"An interface inspired by Ilya Sutskevars'  recommendations.".split(' ').map((word, index) => (
      <motion.span
        key={index}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1}}
        transition={{ delay: index * 0.1, duration: 0.5 }}
        style={{ display: 'inline-block', marginRight: '0.25em' }} // Reduced spacing
      >
        {word}
      </motion.span>
    ))}
  </div>
</div>


< div className='sm:hidden flex items-start w-[100%] sm:w-[50%] mt-5' >
<motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          whileHover={{ scale: 1 }}  
          transition={{ duration: 2 }}
        > <button onClick={scrollToNextSection} className="flex items-start text-xl text-[#1871E3] no-icon  ">
  Check it out
  <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" viewBox="0 0 24 24" fill="currentColor" className="ml-2 mt-[2px]">
              <path d="M13.0001 16.1716L18.3641 10.8076L19.7783 12.2218L12.0001 20L4.22192 12.2218L5.63614 10.8076L11.0001 16.1716V4H13.0001V16.1716Z"></path>
            </svg>
</button></motion.div>
</div  >

</div>
  
</div>
{/* VIDEO */}

{/* <div className='flex flex-col h-screen'>  <!-- Ensures full screen height and flex column layout -->
  <div className='flex-grow'></div>  <!-- Pushes content to center vertically -->
  <div className='text-white text-4xl w-[50%] flex justify-center items-center'>
    Framer components inspired by Dieter Rams' design principles
  </div>
  <div className='flex-grow'></div>  <!-- Balances the space below -->
</div> */}




    <div ref={nextSectionRef} className="flex flex-col items-center  font-Sans min-h-screen relative pt-5">
      
      <div className="absolute  top-5 hidden w-full px-5 sm:flex absolute top-0">
      <Link href="/">
  <div
    className="rounded-lg p-2 transition-colors duration-200 hover:bg-stone-100 sm:bottom-auto hover:bg-[#1871E3]"
    onClick={(e) => {
      window.location.href = '/'; // Set the location to root, which reloads the page
    }}
  >

  </div>
</Link>
        <a
          href="/github"
          target="_blank"
          className="rounded-lg p-2 transition-colors duration-200 hover:bg-stone-100 sm:bottom-auto"
        >
          {/* <GithubIcon /> */}
        </a>
      </div>




      {messages.length > 0 ? (
  <div className="chat-history">
    {messages.map((message, i) => (
      <div key={i} className={clsx("flex w-full items-center justify-center py-2 ", message.role === "user" ? "bg-white" : "")}>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.5 }} className="flex w-full max-w-screen-md items-start space-x-4 px-8 sm:px-0 mt-5">
          {message.role === "assistant" && (
            <div className="p-1.5 text-white bg-[#1871E3]">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="25" height="25">
                <path d="M18 7H22V9H16V3H18V7ZM8 9H2V7H6V3H8V9ZM18 17V21H16V15H22V17H18ZM8 15V21H6V17H2V15H8Z"></path>
              </svg>
            </div>
          )}
          {message.role === "assistant" && (
            <motion.div className="prose mt-2 w-full break-words prose-p:leading-relaxed px-5 font-[300] text-[#919193]">
              {message.content.replace(/^assistant:\s*/, '').split(' ').map((word, index) => (
                <motion.span key={index} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.1 }}>
                  {word + ' '}
                </motion.span>
              ))}
            </motion.div>
          )}
          {message.role === "user" && (
            <div className="flex items-center ml-auto">
              <motion.div className="prose mt-2 w-full break-words prose-p:leading-relaxed px-5 font-[300] text-[#919193] text-right">
                {message.content.replace(/^user:\s*/, '').split(' ').map((word, index) => (
                  <motion.span key={index} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.1 }}>
                    {word + ' '}
                  </motion.span>
                ))}
              </motion.div>
              <div className="p-1.5 text-white">
                <svg xmlns="http://www.w3.org/2000/svg" className="" viewBox="0 0 24 24" fill="#1871E3" width="25" height="25">
                  <path d="M11.9995 2C12.5518 2 12.9995 2.44772 12.9995 3V6C12.9995 6.55228 12.5518 7 11.9995 7C11.4472 7 10.9995 6.55228 10.9995 6V3C10.9995 2.44772 11.4472 2 11.9995 2ZM11.9995 17C12.5518 17 12.9995 17.4477 12.9995 18V21C12.9995 21.5523 12.5518 22 11.9995 22C11.4472 22 10.9995 21.5523 10.9995 21V18C10.9995 17.4477 11.4472 17 11.9995 17ZM20.6597 7C20.9359 7.47829 20.772 8.08988 20.2937 8.36602L17.6956 9.86602C17.2173 10.1422 16.6057 9.97829 16.3296 9.5C16.0535 9.02171 16.2173 8.41012 16.6956 8.13398L19.2937 6.63397C19.772 6.35783 20.3836 6.52171 20.6597 7ZM7.66935 14.5C7.94549 14.9783 7.78161 15.5899 7.30332 15.866L4.70525 17.366C4.22695 17.6422 3.61536 17.4783 3.33922 17C3.06308 16.5217 3.22695 15.9101 3.70525 15.634L6.30332 14.134C6.78161 13.8578 7.3932 14.0217 7.66935 14.5ZM20.6597 17C20.3836 17.4783 19.772 17.6422 19.2937 17.366L16.6956 15.866C16.2173 15.5899 16.0535 14.9783 16.3296 14.5C16.6057 14.0217 17.2173 13.8578 17.6956 14.134L20.2937 15.634C20.772 15.9101 20.9359 16.5217 20.6597 17ZM7.66935 9.5C7.3932 9.97829 6.78161 10.1422 6.30332 9.86602L3.70525 8.36602C3.22695 8.08988 3.06308 7.47829 3.33922 7C3.61536 6.52171 4.22695 6.35783 4.70525 6.63397L7.30332 8.13398C7.78161 8.41012 7.94549 9.02171 7.66935 9.5Z"></path>
                </svg>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    ))}
  </div>
) : 
 (
        <div className="mx-8 mt-0 max-w-screen-md sm:w-full font-Sans">
  <motion.div
  variants={containerVariants}
  initial="hidden"
  animate="visible"

  transition={{ borderColor: { duration: 2 } }} // Animate to visible border color
  className="flex flex-col space-y-4 sm:p-5 mt-10"
>
        <motion.h1        variants={childVariants}>
          Ilya's Papers
        </motion.h1>
        <motion.h3 variants={childVariants} className='max-w-2xl'>
  This is an open-source AI chat interface that uses OpenAI Functions and Vercel AI SDK to interact with Ilya Sutskevar's list of research recommendations with natural language.
</motion.h3>

<motion.div variants={childVariants} className="flex flex-row sm:flex-row sm:space-x-2 space-x-1 mt-4 font-[300] items-center sm:items-start">
  <div className="text-grey mb-2 sm:mb-0 mr-2 sm:mr-0">Explore</div>
  <div className="px-2 mb-2 border border-[#1871E3] text-[#1871E3] rounded-full hover:bg-[#1871E3] hover:text-white">DL</div>
  <div className="px-2 mb-2 border border-[#1871E3] text-[#1871E3] rounded-full hover:bg-[#1871E3] hover:text-white">Transformers</div>
  <div className="px-2 mb-2 border border-[#1871E3] text-[#1871E3] rounded-full hover:bg-[#1871E3] hover:text-white">RNNs</div>
</motion.div>




        <motion.div
  variants={containerVariants}
  initial="hidden"
  animate="visible"
  style={{
    position: "relative", // Ensure the container is positioned to hold absolute children
    borderTop: "1px solid transparent" // Set the border but make it transparent
  }}
  className="flex flex-col space-y-0 border-t border-gray-200 pt-5 "
>
  <motion.div
    initial={{ width: "0%" }} // Start with no visible border
    animate={{ width: "100%" }} // Animate to full width
    transition={{ duration:1 }}
    style={{
      position: "absolute",
      top: 0, // Align to the top of the parent
      left: 0, // Start from the left
      height: "1px", // Border thickness
      backgroundColor: "rgba(200, 200, 200, 1)", // Visible border color
    }}
  />
          {examples.map((example, i) => (
            <motion.button
              key={i}
              variants={childVariants}
         
              onClick={() => {
                setInput(example);
                inputRef.current?.focus();
              }}
              className="  mt-5"
            >
              {example}
            </motion.button>
          ))}
        </motion.div>
      </motion.div>
      </div>
      )}
      <div className="flex w-full flex-col  bottom-0 justify-end items-center  space-y-3 p-5 pb-3  px-8 sm:px-0 ">
        <form
          ref={formRef}
          onSubmit={handleSubmit}
          className="absolute bottom-0 mb-10  w-11/12 md:max-w-3xl rounded-xl bg-[#E7E7E8] px-4 pb-2 pt-3 sm:pb-3 sm:pt-4"
        >
          <Textarea
            ref={inputRef}
            tabIndex={0}
            required
            rows={1}
        
            placeholder="Send a message"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                formRef.current?.requestSubmit();
                e.preventDefault();
              }
            }}
            spellCheck={false}
            className="w-full pr-10 focus:outline-none bg-[#E7E7E8] resize-none overflow-hidden textarea-custom"
          />
      <button
  className={clsx(
    "absolute inset-y-0 right-3 my-auto flex h-8 w-8 items-center justify-center rounded-md transition-all no-hover-icon",
    disabled
      ? "cursor-not-allowed"
      : "hover:bg-[#1871E3] hover-icon-white",
  )}
  disabled={disabled}
>
  {isLoading ? (
    <LoadingCircle />
  ) : (
    <img
      src="data:image/svg+xml,%3csvg stroke-width='2' id='Layer_1' data-name='Layer 1' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3e%3cdefs%3e%3cstyle%3e.cls-zs13fqo78bh100s320feaon-1%7bfill:none%3bstroke:%231871E3%3bstroke-miterlimit:10%3b%3b%7d%3c/style%3e%3c/defs%3e%3cline class='cls-zs13fqo78bh100s320feaon-1' x1='21.5' y1='12' x2='0.5' y2='12'/%3e%3cpolyline class='cls-zs13fqo78bh100s320feaon-1' points='13.86 4.36 21.5 12 13.86 19.64'/%3e%3c/svg%3e"
      className={clsx(
        "h-4 w-4",
        input.length === 0 ? "text-gray-300" : "text-white",
      )}
      alt="Send Icon"
    />
  )}
</button>
        </form>
        <p className="text-center text-xs text-gray-400 absolute bottom-3">
          Built with{" "} love from{" "}
          <a
            href="https://platform.openai.com/docs/guides/gpt/function-calling"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-colors hover:text-black"
          >
           @alexdphan
          </a>{" "}
          and{" "}
          <a
            href="https://sdk.vercel.ai/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-colors hover:text-black"
          >
           @willdphan
          </a>
       
        </p>
      </div>
      </div>
   
  
    </main>
  );
      }