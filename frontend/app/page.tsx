"use client";
import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer,
  Tooltip, XAxis, YAxis
} from "recharts";
import {
  Activity, ArrowDownToLine, ArrowRight, BarChart3, BrainCircuit,
  CarFront, CheckCircle2, ChevronDown, CircleDollarSign, Database,
  FileSpreadsheet, Gauge, Github, Layers3, LoaderCircle, Menu,
  Network, RefreshCw, ShieldCheck, Sparkles, UploadCloud, Zap
} from "lucide-react";

type Metrics = { model:string; r2:number; mae:number; rmse:number };
type Meta = {
  features:string[]; model_name:string; metrics:Metrics[]; brands:string[];
  models_by_brand:Record<string,string[]>; seller_types:string[];
  fuel_types:string[]; transmission_types:string[]; seats:number[]; dataset_rows:number;
};
type Vehicle = {
  brand:string; model:string; seller_type:string; fuel_type:string;
  transmission_type:string; vehicle_age:number; km_driven:number;
  mileage:number; engine:number; max_power:number; seats:number;
};
type Prediction = { prediction:number; currency:string; model:string };
type Explain = { base_value:number; prediction:number; features:{feature:string;impact:number}[]; interpretation:string };

const API = (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "");
const fmt = (v:number) => new Intl.NumberFormat("en-IN",{style:"currency",currency:"INR",maximumFractionDigits:0}).format(v);
const pct = (v:number) => `${(v*100).toFixed(2)}%`;

const defaults:Vehicle = {
  brand:"Maruti",model:"Swift",seller_type:"Individual",fuel_type:"Petrol",
  transmission_type:"Manual",vehicle_age:5,km_driven:45000,mileage:20,
  engine:1197,max_power:88.5,seats:5
};

function SelectField({label,value,options,onChange}:{label:string;value:string|number;options:(string|number)[];onChange:(v:string)=>void}) {
  return <label className="field"><span>{label}</span><div className="select-wrap"><select value={value} onChange={e=>onChange(e.target.value)}>{options.map(o=><option key={o} value={o}>{o}</option>)}</select><ChevronDown size={15}/></div></label>;
}
function NumberField({label,value,onChange,min=0,max=1000000,step=1,suffix=""}:{label:string;value:number;onChange:(v:number)=>void;min?:number;max?:number;step?:number;suffix?:string}) {
  return <label className="field"><span>{label}</span><div className="number-wrap"><input type="number" min={min} max={max} step={step} value={value} onChange={e=>onChange(Number(e.target.value))}/>{suffix&&<small>{suffix}</small>}</div></label>;
}
function Stat({label,value,sub,icon:Icon}:{label:string;value:string;sub:string;icon:any}) {
  return <div className="stat-card"><div className="stat-top"><span>{label}</span><Icon size={18}/></div><strong>{value}</strong><small>{sub}</small></div>;
}

export default function Home() {
  const [meta,setMeta]=useState<Meta|null>(null);
  const [vehicle,setVehicle]=useState<Vehicle>(defaults);
  const [prediction,setPrediction]=useState<Prediction|null>(null);
  const [explain,setExplain]=useState<Explain|null>(null);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState("");
  const [tab,setTab]=useState("estimate");
  const [batchRows,setBatchRows]=useState<Vehicle[]>([]);
  const [batchResults,setBatchResults]=useState<any[]>([]);
  const [batchName,setBatchName]=useState("");
  const [apiState,setApiState]=useState<"checking"|"online"|"offline">("checking");

  useEffect(()=>{
    if(!API){setApiState("offline");setError("Set NEXT_PUBLIC_API_BASE_URL to your deployed Flask API URL.");return;}
    fetch(`${API}/metadata`).then(async r=>{if(!r.ok)throw new Error("API unavailable");return r.json();})
      .then((m:Meta)=>{setMeta(m);setApiState("online");setVehicle(v=>({...v,brand:m.brands.includes(v.brand)?v.brand:m.brands[0],model:m.models_by_brand[m.brands[0]]?.[0]||v.model}));})
      .catch(()=>{setApiState("offline");setError("Could not connect to the valuation API. Check the backend deployment and CORS_ORIGINS.");});
  },[]);

  const models = useMemo(()=>meta?.models_by_brand[vehicle.brand]||[],[meta,vehicle.brand]);
  const set = (key:keyof Vehicle,value:any) => setVehicle(v=>({...v,[key]:value,...(key==="brand"?{model:meta?.models_by_brand[String(value)]?.[0]||""}:{})}));

  async function post(path:string,body:any) {
    const response=await fetch(`${API}${path}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||"Request failed");
    return data;
  }
  async function estimate() {
    setBusy(true);setError("");setPrediction(null);setExplain(null);
    try {
      const result=await post("/predict",{record:vehicle});
      setPrediction(result);
      const detail=await post("/explain",{record:vehicle});
      setExplain(detail);
    } catch(e:any){setError(e.message||"Prediction failed");}
    finally{setBusy(false);}
  }
  async function runBatch() {
    setBusy(true);setError("");
    try {
      const result=await post("/batch-predict",{records:batchRows});
      setBatchResults(result.results);
    } catch(e:any){setError(e.message||"Batch prediction failed");}
    finally{setBusy(false);}
  }
  function parseCsv(file:File) {
    setBatchName(file.name);setBatchResults([]);
    Papa.parse(file,{header:true,skipEmptyLines:true,dynamicTyping:true,complete:(res)=>{
      const records=res.data as any[];
      const required=meta?.features||[];
      const missing=required.filter(k=>!res.meta.fields?.includes(k));
      if(missing.length){setError(`CSV is missing columns: ${missing.join(", ")}`);return;}
      setError("");setBatchRows(records as Vehicle[]);
    },error:()=>setError("Unable to read this CSV file.")});
  }
  function downloadCsv() {
    if(!batchResults.length)return;
    const csv=Papa.unparse(batchResults);
    const url=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
    const a=document.createElement("a");a.href=url;a.download="carinsight-valuation-report.csv";a.click();URL.revokeObjectURL(url);
  }

  const best=meta?.metrics?.[0];
  const sortedMetrics=[...(meta?.metrics||[])].sort((a,b)=>b.r2-a.r2);
  const demoChart=sortedMetrics.map(m=>({name:m.model.replace(" ","\n"),score:Number((m.r2*100).toFixed(2))}));

  return <main className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><CarFront size={23}/></div><div><b>CarInsight<span> Pro</span></b><small>VEHICLE INTELLIGENCE</small></div></div>
      <div className="side-label">WORKSPACE</div>
      <button className={`nav-item ${tab==="estimate"?"active":""}`} onClick={()=>setTab("estimate")}><Gauge size={17}/> Valuation studio</button>
      <button className={`nav-item ${tab==="batch"?"active":""}`} onClick={()=>setTab("batch")}><FileSpreadsheet size={17}/> Batch valuation</button>
      <button className={`nav-item ${tab==="models"?"active":""}`} onClick={()=>setTab("models")}><BrainCircuit size={17}/> Model intelligence</button>
      <div className="side-bottom">
        <div className="api-status"><span className={`pulse ${apiState}`}/><div><b>API {apiState==="online"?"Operational":apiState==="offline"?"Disconnected":"Connecting"}</b><small>Flask inference service</small></div></div>
        <div className="side-foot">CARINSIGHT PRO <span>v1.0.0</span></div>
      </div>
    </aside>

    <section className="main">
      <header className="topbar"><div><span className="crumb">PLATFORM</span><span className="slash">/</span><b>{tab==="estimate"?"Valuation Studio":tab==="batch"?"Batch Valuation":"Model Intelligence"}</b></div><div className="top-actions"><span className="live-dot"/> LIVE SYSTEM <a href="https://github.com/vedantj27/CarInsight-Pro" target="_blank" rel="noreferrer" aria-label="GitHub"><Github size={18}/></a></div></header>

      <section className="hero">
        <div className="hero-copy"><div className="eyebrow"><Sparkles size={13}/> INTELLIGENT AUTOMOTIVE VALUATION</div>
          <h1>Know the value.<br/><em>Drive the decision.</em></h1>
          <p>AI-powered vehicle valuation backed by XGBoost, market analytics and transparent model explanations.</p>
          <div className="hero-tags"><span><Zap size={13}/> Real-time inference</span><span><ShieldCheck size={13}/> Explainable AI</span><span><Database size={13}/> CarDekho dataset</span></div>
        </div>
        <div className="hero-visual"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="car-glow"><CarFront size={92} strokeWidth={1.1}/></div><div className="visual-chip chip-top"><Activity size={14}/> MODEL ACTIVE</div><div className="visual-chip chip-bottom"><span>R²</span> {best?pct(best.r2):"—"} <small>TEST SCORE</small></div></div>
      </section>

      <section className="stats-grid">
        <Stat label="Selected algorithm" value={meta?.model_name||"Loading"} sub="Best held-out R² score" icon={BrainCircuit}/>
        <Stat label="Test R² score" value={best?pct(best.r2):"—"} sub="80/20 holdout evaluation" icon={Activity}/>
        <Stat label="Mean absolute error" value={best?fmt(best.mae):"—"} sub="Average test-set error" icon={CircleDollarSign}/>
        <Stat label="Training records" value={meta?meta.dataset_rows.toLocaleString("en-IN"):"—"} sub="CarDekho vehicle listings" icon={Database}/>
      </section>

      {error&&<div className="alert"><span>!</span>{error}<button onClick={()=>setError("")}>Dismiss</button></div>}

      {tab==="estimate"&&<section className="content-grid">
        <div className="panel form-panel">
          <div className="panel-head"><div><div className="section-kicker">VALUATION INPUT</div><h2>Vehicle specification</h2><p>Enter the vehicle details to generate a market estimate.</p></div><span className="step-pill">01 / INPUT</span></div>
          <div className="form-section"><div className="form-title"><span>01</span> Vehicle identity</div><div className="fields three">
            <SelectField label="Brand" value={vehicle.brand} options={meta?.brands||[vehicle.brand]} onChange={v=>set("brand",v)}/>
            <SelectField label="Model" value={vehicle.model} options={models.length?models:[vehicle.model]} onChange={v=>set("model",v)}/>
            <SelectField label="Fuel type" value={vehicle.fuel_type} options={meta?.fuel_types||[vehicle.fuel_type]} onChange={v=>set("fuel_type",v)}/>
            <SelectField label="Transmission" value={vehicle.transmission_type} options={meta?.transmission_types||[vehicle.transmission_type]} onChange={v=>set("transmission_type",v)}/>
            <SelectField label="Seller type" value={vehicle.seller_type} options={meta?.seller_types||[vehicle.seller_type]} onChange={v=>set("seller_type",v)}/>
            <SelectField label="Seats" value={vehicle.seats} options={meta?.seats||[vehicle.seats]} onChange={v=>set("seats",Number(v))}/>
          </div></div>
          <div className="form-section"><div className="form-title"><span>02</span> Usage profile</div><div className="fields three">
            <NumberField label="Vehicle age" value={vehicle.vehicle_age} min={0} max={35} onChange={v=>set("vehicle_age",v)} suffix="years"/>
            <NumberField label="Kilometres driven" value={vehicle.km_driven} min={0} max={1000000} step={1000} onChange={v=>set("km_driven",v)} suffix="km"/>
            <NumberField label="Mileage" value={vehicle.mileage} min={1} max={60} step={0.1} onChange={v=>set("mileage",v)} suffix="km/l"/>
          </div></div>
          <div className="form-section"><div className="form-title"><span>03</span> Engine specifications</div><div className="fields three">
            <NumberField label="Engine capacity" value={vehicle.engine} min={400} max={8000} step={50} onChange={v=>set("engine",v)} suffix="cc"/>
            <NumberField label="Maximum power" value={vehicle.max_power} min={10} max={1000} step={0.5} onChange={v=>set("max_power",v)} suffix="bhp"/>
          </div></div>
          <button className="primary-btn" onClick={estimate} disabled={busy||apiState!=="online"}>{busy?<LoaderCircle className="spin" size={18}/>:<Sparkles size={17}/>} {busy?"Analysing vehicle...":"Generate valuation"} <ArrowRight size={17}/></button>
          <div className="fine-print"><ShieldCheck size={13}/> Your inputs are processed by the CarInsight inference API.</div>
        </div>

        <div className="right-stack">
          <div className="panel result-panel">
            <div className="panel-head"><div><div className="section-kicker">VALUATION OUTPUT</div><h2>Estimated market value</h2></div><span className="result-icon"><CircleDollarSign size={20}/></span></div>
            {prediction?<><div className="result-value">{fmt(prediction.prediction)}</div><div className="result-caption">Estimated resale value · Indian Rupee</div><div className="result-divider"/><div className="result-meta"><span>Inference model</span><b>{prediction.model}</b></div><div className="result-meta"><span>Valuation status</span><b className="green">Completed</b></div></>:<div className="empty-result"><div className="empty-icon"><CarFront size={30}/></div><b>Your valuation appears here</b><span>Complete the vehicle details and generate an estimate.</span></div>}
            <div className="disclaimer">Indicative model estimate only. Actual resale value depends on condition, ownership, location and market demand.</div>
          </div>
          <div className="panel explain-panel"><div className="panel-head"><div><div className="section-kicker">MODEL TRANSPARENCY</div><h2>SHAP feature impact</h2></div><span className="shap-badge">SHAP</span></div>
            {explain?<><p className="muted small">{explain.interpretation}</p><div className="shap-list">{explain.features.slice(0,7).map(f=><div className="shap-row" key={f.feature}><span title={f.feature}>{f.feature.replace("categorical__","").replace("numeric__","")}</span><div className="shap-track"><i className={f.impact>=0?"positive":"negative"} style={{width:`${Math.min(100,Math.abs(f.impact)/(Math.max(...explain.features.map(x=>Math.abs(x.impact)),1))*100)}%`}}/></div><b className={f.impact>=0?"positive-text":"negative-text"}>{f.impact>=0?"+":"−"}{fmt(Math.abs(f.impact))}</b></div>)}</div><div className="shap-legend"><span><i className="pos-dot"/> Raises estimate</span><span><i className="neg-dot"/> Lowers estimate</span></div></>:<div className="empty-explain"><Network size={26}/><span>Generate a valuation to see which vehicle attributes influence the estimate.</span></div>}
          </div>
        </div>
      </section>}

      {tab==="batch"&&<section className="panel batch-panel"><div className="panel-head"><div><div className="section-kicker">BULK WORKFLOW</div><h2>Batch vehicle valuation</h2><p>Upload a CSV with one vehicle per row. The API returns a downloadable valuation report.</p></div><span className="step-pill">02 / BATCH</span></div>
        <div className="upload-zone"><UploadCloud size={34}/><h3>Upload vehicle dataset</h3><p>{batchName||"CSV file containing all required vehicle input columns"}</p><label className="upload-btn">Choose CSV<input type="file" accept=".csv,text/csv" onChange={e=>e.target.files?.[0]&&parseCsv(e.target.files[0])}/></label><small>Required: brand, model, seller_type, fuel_type, transmission_type, vehicle_age, km_driven, mileage, engine, max_power, seats</small></div>
        {batchRows.length>0&&<div className="batch-summary"><span>{batchRows.length.toLocaleString()} vehicles loaded</span><button className="primary-btn compact" onClick={runBatch} disabled={busy}>{busy?<LoaderCircle className="spin" size={16}/>:<Zap size={16}/>} Run bulk valuation</button></div>}
        {batchResults.length>0&&<><div className="table-wrap"><table><thead><tr><th>Brand</th><th>Model</th><th>Fuel</th><th>Age</th><th>Predicted price</th></tr></thead><tbody>{batchResults.slice(0,20).map((r,i)=><tr key={i}><td>{r.brand}</td><td>{r.model}</td><td>{r.fuel_type}</td><td>{r.vehicle_age} yrs</td><td className="table-price">{fmt(r.predicted_price)}</td></tr>)}</tbody></table></div><button className="secondary-btn" onClick={downloadCsv}><ArrowDownToLine size={16}/> Download valuation report CSV</button></>}
      </section>}

      {tab==="models"&&<section className="model-grid">
        <div className="panel benchmark"><div className="panel-head"><div><div className="section-kicker">MODEL BENCHMARK</div><h2>Regression performance</h2><p>Four algorithms evaluated on the same holdout split.</p></div><span className="step-pill">03 / ML</span></div>
          <div className="chart"><ResponsiveContainer width="100%" height={270}><BarChart data={demoChart} margin={{top:12,right:8,left:0,bottom:4}}><CartesianGrid strokeDasharray="3 3" stroke="#25314a"/><XAxis dataKey="name" tick={{fill:"#8e9ab3",fontSize:11}} axisLine={false} tickLine={false}/><YAxis domain={[0,100]} tick={{fill:"#8e9ab3",fontSize:11}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#111b2e",border:"1px solid #293957",borderRadius:10}} formatter={(v:any)=>[`${v}%`,"R² score"]}/><Bar dataKey="score" fill="#8c7bff" radius={[7,7,0,0]}/></BarChart></ResponsiveContainer></div>
          <div className="metrics-table">{sortedMetrics.map((m,i)=><div className="metric-line" key={m.model}><span className="rank">{String(i+1).padStart(2,"0")}</span><b>{m.model}</b>{m.model===meta?.model_name&&<span className="winner">SELECTED</span>}<strong>{pct(m.r2)}</strong></div>)}</div>
        </div>
        <div className="panel model-notes"><div className="section-kicker">MODEL INFORMATION</div><h2><BrainCircuit size={22}/> {meta?.model_name||"Loading model"}</h2><p>The selected regressor is chosen from the held-out evaluation by the highest R² score. SHAP values expose the direction and magnitude of feature contributions.</p><div className="note"><Layers3 size={18}/><div><b>Preprocessing pipeline</b><span>Numeric imputation and one-hot categorical encoding.</span></div></div><div className="note"><Sparkles size={18}/><div><b>Explainability</b><span>SHAP TreeExplainer for individual prediction impact.</span></div></div><div className="note"><ShieldCheck size={18}/><div><b>Evaluation discipline</b><span>Same train/test split and metrics across candidates.</span></div></div></div>
      </section>}

      <footer className="footer"><span>© 2026 CarInsight Pro · AI Vehicle Valuation System</span><span>Built with <b>Python</b>, <b>XGBoost</b>, <b>Flask</b>, <b>Next.js</b> & <b>SHAP</b></span></footer>
    </section>
  </main>;
}
