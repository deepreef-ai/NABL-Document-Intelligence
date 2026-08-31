import { Navigate, Route, Routes } from "react-router-dom";
import "./App.css";
import ReviewPage from "./pages/ReviewPage";
import UploadPage from "./pages/UploadPage";
import WizardPage from "./pages/WizardPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<WizardPage />} />
      <Route path="/applications/:applicationId/wizard" element={<WizardPage />} />
      <Route path="/applications/:applicationId/upload" element={<UploadPage />} />
      <Route path="/applications/:applicationId/review" element={<ReviewPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
