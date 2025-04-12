import{_ as t,c as a,o as s,aj as e}from"./chunks/framework.DH2d4kZa.js";const c=JSON.parse('{"title":"Introduction","description":"","frontmatter":{"layout":"home","hero":{"name":"QuboSolver.jl","tagline":"A Julia suite implementing the GCS algorithm and other heuristics for solving QUBO problems","actions":[{"theme":"brand","text":"Read the preprint","link":"https://arxiv.org/abs/2501.09078"},{"theme":"alt","text":"Getting Started","link":"/getting_started"},{"theme":"alt","text":"API","link":"/resources/api"},{"theme":"alt","text":"View on Github","link":"https://github.com/LorenzoFioroni/QuboSolver.jl"}]}},"headers":[],"relativePath":"index.md","filePath":"index.md","lastUpdated":null}'),n={name:"index.md"};function o(l,i,r,h,p,d){return s(),a("div",null,i[0]||(i[0]=[e(`<h1 id="doc:Introduction" tabindex="-1">Introduction <a class="header-anchor" href="#doc:Introduction" aria-label="Permalink to &quot;Introduction {#doc:Introduction}&quot;">​</a></h1><p><a href="https://github.com/LorenzoFioroni/QuboSolver.jl" target="_blank" rel="noreferrer"><code>QuboSolver.jl</code></a> is a <a href="https://julialang.org/" target="_blank" rel="noreferrer"><code>Julia</code></a> package that provides a suite of tools for solving <em>Quadratic Unconstrained Binary Optimization</em> (QUBO) problems. Importantly, it implements the <strong>GCS</strong> algorithm which we propose in our preprint &quot;<a href="https://arxiv.org/abs/2501.09078" target="_blank" rel="noreferrer">Entanglement-assisted heuristic for variational solutions of discrete optimization problems</a>&quot;. Additionally, it includes a variety of other solvers and utilities for benchmarking and comparing different approaches to QUBO optimization.</p><h1 id="doc:Installation" tabindex="-1">Installation <a class="header-anchor" href="#doc:Installation" aria-label="Permalink to &quot;Installation {#doc:Installation}&quot;">​</a></h1><div class="tip custom-block"><p class="custom-block-title">Requirements</p><p><code>QuboSolver.jl</code> requires <code>Julia 1.10+</code>.</p></div><p>To install <code>QuboSolver.jl</code>, run the following commands inside Julia&#39;s interactive session (REPL):</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Pkg</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(url</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;https://github.com/LorenzoFioroni/QuboSolver.jl&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Alternatively, this can also be run in <code>Julia</code>&#39;s <a href="https://julialang.github.io/Pkg.jl/v1/getting-started/" target="_blank" rel="noreferrer">Pkg REPL</a> by pressing the key <code>]</code> in the REPL and running:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) pkg</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> add https</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">://</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">github</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">com</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">LorenzoFioroni</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">QuboSolver</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">jl</span></span></code></pre></div><p>Finally, to start using the package execute the following line of code:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> QuboSolver</span></span></code></pre></div><p>The package is now ready to be used. You can start by checking the <a href="./getting_started">Getting Started</a> page for a quick introduction to the package and its features. You can also check the <a href="./resources/api">API documentation</a> for a more detailed overview of the available functions and their usage.</p>`,11)]))}const g=t(n,[["render",o]]);export{c as __pageData,g as default};
