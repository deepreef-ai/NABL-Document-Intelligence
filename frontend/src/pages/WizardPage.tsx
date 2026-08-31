import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import ChatPanel from "../components/ChatPanel";
import {
  answerWizardQuestion,
  createApplication,
  getWizardState,
  type Application,
  type ChatEntry,
  type EligibilityState,
  type NablFormType,
} from "../api/client";

const FORM_GROUPS: { title: string; options: { value: NablFormType; label: string }[] }[] = [
  {
    title: "Full accreditation",
    options: [
      { value: "NABL_151", label: "NABL 151 — Testing Laboratory" },
      { value: "NABL_152", label: "NABL 152 — Calibration Laboratory" },
      { value: "NABL_153", label: "NABL 153 — Medical Laboratory (ISO 15189)" },
      { value: "NABL_153A", label: "NABL 153A — Medical Lab (Operational/Supporting Entities)" },
      { value: "NABL_154", label: "NABL 154 — Testing Lab + Regulatory Recognition (FSSAI/APEDA/etc.)" },
      { value: "NABL_158", label: "NABL 158 — Product-Based Testing Laboratory" },
    ],
  },
  {
    title: "Recognition scheme",
    options: [
      { value: "NABL_155", label: "NABL 155 — Medical (Entry Level) Testing Lab" },
      { value: "NABL_157", label: "NABL 157 — Government Drinking Water Testing Lab" },
      { value: "NABL_159", label: "NABL 159 — Temporary Site Lab (Aggregates/Concrete)" },
    ],
  },
];

export default function WizardPage() {
  const { applicationId } = useParams();
  const navigate = useNavigate();
  const [application, setApplication] = useState<Application | null>(null);
  const [state, setState] = useState<EligibilityState | null>(null);
  const [history, setHistory] = useState<ChatEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!applicationId) return;
    getWizardState(applicationId)
      .then((res) => {
        setApplication(res.application);
        setState(res.state);
        setHistory(res.history);
      })
      .catch((e) => setError(String(e)));
  }, [applicationId]);

  async function handlePickFormType(formType: NablFormType) {
    setError(null);
    try {
      const res = await createApplication(formType);
      navigate(`/applications/${res.application.id}/wizard`);
    } catch (e) {
      setError(String(e));
    }
  }

  async function handleSend(message: string) {
    if (!applicationId) return;
    setHistory((h) => [...h, { role: "user", content: message }]);
    const res = await answerWizardQuestion(applicationId, message);
    setApplication(res.application);
    setHistory((h) => [...h, { role: "assistant", content: res.reply }]);
    const refreshed = await getWizardState(applicationId);
    setState(refreshed.state);
  }

  if (!applicationId) {
    return (
      <div className="page wizard-page">
        <h1>NABL Document Intelligence</h1>
        <p>Which accreditation are you applying for?</p>
        {FORM_GROUPS.map((group) => (
          <div key={group.title} className="form-type-group">
            <h2>{group.title}</h2>
            <div className="form-type-picker">
              {group.options.map((f) => (
                <button key={f.value} onClick={() => handlePickFormType(f.value)}>
                  {f.label}
                </button>
              ))}
            </div>
          </div>
        ))}
        {error && <p className="error">{error}</p>}
      </div>
    );
  }

  return (
    <div className="page wizard-page">
      <h1>Eligibility check — {application?.form_type}</h1>
      {state && (
        <ul className="prerequisite-checklist">
          {Object.values(state.answers).map((a) => (
            <li key={a.question_id} className={a.satisfied === true ? "done" : a.satisfied === false ? "blocked" : ""}>
              {a.satisfied === true ? "✔" : a.satisfied === false ? "✘" : "…"} {a.question_id.replace(/_/g, " ")}
            </li>
          ))}
        </ul>
      )}
      <ChatPanel history={history} onSend={handleSend} disabled={state?.all_satisfied} />
      {state?.all_satisfied && (
        <button className="primary" onClick={() => navigate(`/applications/${applicationId}/upload`)}>
          Continue to document upload →
        </button>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}
