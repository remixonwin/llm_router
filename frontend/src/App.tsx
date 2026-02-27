import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Layout } from "./components/Layout/Layout";
import { Dashboard } from "./components/Dashboard/Dashboard";
import { Models } from "./components/Models/Models";
import { Stats } from "./components/Stats/Stats";
import { Admin } from "./components/Admin/Admin";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10000,
      retry: 1,
    },
  },
});

// Get basename from env or default to "/" for container deployment
// In unified gateway, this would be "/llm"
const BASENAME = import.meta.env.VITE_ROUTER_BASENAME || "/";

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter basename={BASENAME}>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<Models />} />
            <Route path="/stats" element={<Stats />} />
            <Route path="/admin" element={<Admin />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
