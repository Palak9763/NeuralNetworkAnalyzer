import React from "react";
import ReactDOM from "react-dom/client";
import { GoogleOAuthProvider } from "@react-oauth/google";
import { ThemeProvider } from "./components/ThemeContext";
import App from "./App";
import "./index.css";

console.log(import.meta.env);
console.log(import.meta.env.VITE_GOOGLE_CLIENT_ID);

let GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID ?? "";
if (!GOOGLE_CLIENT_ID || GOOGLE_CLIENT_ID === "your-google-client-id-here") {
  // Use a placeholder format that doesn't trigger GIS validation crash on load
  GOOGLE_CLIENT_ID = "100000000000-placeholder.apps.googleusercontent.com";
}


ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </GoogleOAuthProvider>
  </React.StrictMode>
);
