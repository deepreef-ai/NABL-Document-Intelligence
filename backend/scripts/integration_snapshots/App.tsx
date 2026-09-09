import { Navigate, Route, Routes } from "react-router-dom";
import "./App.css";
import ReviewPage from "./pages/ReviewPage";
import UploadPage from "./pages/UploadPage";

/**
 * Two screens: upload, then review.
 *
 * There is no wizard and no start step. "/" lands straight on the upload box,
 * which silently acquires an application of its own — the eligibility
 * questions were bookkeeping the user did not ask for, and every route that
 * led to them has been removed rather than merely hidden, so nothing can
 * navigate back into a flow that no longer exists.
 */
function App() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/applications/:applicationId/upload" element={<UploadPage />} />
      <Route path="/applications/:applicationId/review" element={<ReviewPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
