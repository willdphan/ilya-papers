export const runtime = "edge";

export async function POST(req: Request) {
  try {
    const { input } = await req.json();

    // Fetch response from local server using IPv4 address
    const response = await fetch('http://127.0.0.1:8000/generate_response', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input }),
    });

    if (!response.ok) {
      throw new Error(`Error fetching data from local server: ${response.statusText}`);
    }

    const initialResponse = await response.json();

    return new Response(JSON.stringify(initialResponse), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  } catch (error) {
    console.error('Fetch error:', error);
    return new Response("Error fetching data from local server", {
      status: 500,
    });
  }
}