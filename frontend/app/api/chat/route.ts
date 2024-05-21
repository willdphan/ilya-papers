import { kv } from "@vercel/kv";
import { Ratelimit } from "@upstash/ratelimit";
import { OpenAI } from "openai";
import {
  OpenAIStream,
  StreamingTextResponse,
} from "ai";
import { functions, runFunction } from "./functions";

// Create an OpenAI API client (that's edge friendly!)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export const runtime = "edge";

export async function POST(req: Request) {
  if (
    process.env.NODE_ENV !== "development" &&
    process.env.KV_REST_API_URL &&
    process.env.KV_REST_API_TOKEN
  ) {
    const ip = req.headers.get("x-forwarded-for");
    const ratelimit = new Ratelimit({
      redis: kv,
      limiter: Ratelimit.slidingWindow(50, "1 d"),
    });

    const { success, limit, reset, remaining } = await ratelimit.limit(
      `chathn_ratelimit_${ip}`,
    );

    if (!success) {
      return new Response("You have reached your request limit for the day.", {
        status: 429,
        headers: {
          "X-RateLimit-Limit": limit.toString(),
          "X-RateLimit-Remaining": remaining.toString(),
          "X-RateLimit-Reset": reset.toString(),
        },
      });
    }
  }

  const { messages } = await req.json();

  try {
    // Prepare the input for the local server
    const inputText = messages.map(msg => msg.content).join('\n');

    // Fetch response from local server using IPv4 address
    const response = await fetch('http://127.0.0.1:8000/generate_response', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input: inputText }),
    });

    if (!response.ok) {
      throw new Error(`Error fetching data from local server: ${response.statusText}`);
    }

    const initialResponse = await response.json();

    const stream = OpenAIStream(initialResponse.response, {
      experimental_onFunctionCall: async (
        { name, arguments: args },
        createFunctionCallMessages,
      ) => {
        const result = await runFunction(name, args);
        const newMessages = createFunctionCallMessages(result);
        return openai.chat.completions.create({
          model: "gpt-3.5-turbo-0613",
          stream: true,
          messages: [...messages, ...newMessages],
        });
      },
    });

    return new StreamingTextResponse(stream);
  } catch (error) {
    console.error('Fetch error:', error);
    return new Response("Error fetching data from local server", {
      status: 500,
    });
  }
}