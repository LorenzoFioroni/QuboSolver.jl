import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";

const baseTemp = {
    base: '/QuboSolver.jl/previews/PR3/',
}

const navTemp = {
  nav: [
{ text: 'Home', link: '/index' },
{ text: 'Getting Started', link: '/getting_started' },
{ text: 'Resources', collapsed: false, items: [
{ text: 'API', link: '/resources/api' },
{ text: 'Bibliography', link: '/resources/bibliography' },
{ text: 'Citing', link: '/resources/citing' }]
 }
]
,
}

const nav = [
  ...navTemp.nav,
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
    base: baseTemp.base,
    title: 'QuboSolver.jl',
    description: 'Documentation for QuboSolver.jl',
    lastUpdated: true,
    cleanUrls: true,
    outDir: '../1', // This is required for MarkdownVitepress to work correctly...
    head: [
        
    ],
    ignoreDeadLinks: true,

    markdown: {
        math: true,

        // options for @mdit-vue/plugin-toc
        // https://github.com/mdit-vue/mdit-vue/tree/main/packages/plugin-toc#options
        toc: { level: [2, 3, 4] }, // for API page, triggered by: [[toc]]

        config(md) {
            md.use(tabsMarkdownPlugin),
                md.use(mathjax3),
                md.use(footnote)
        },
        theme: {
            light: "github-light",
            dark: "github-dark"
        }
    },
    themeConfig: {
        outline: 'deep',
        
        search: {
            provider: 'local',
            options: {
                detailedView: true
            }
        },
        nav,
        sidebar: [
{ text: 'Home', link: '/index' },
{ text: 'Getting Started', link: '/getting_started' },
{ text: 'Resources', collapsed: false, items: [
{ text: 'API', link: '/resources/api' },
{ text: 'Bibliography', link: '/resources/bibliography' },
{ text: 'Citing', link: '/resources/citing' }]
 }
]
,
        editLink: { pattern: "https://github.com/LorenzoFioroni/QuboSolver.jl/edit/main/docs/src/:path" },
        socialLinks: [
            { icon: 'github', link: 'https://github.com/LorenzoFioroni/QuboSolver.jl' }
        ],
        footer: {
            message: 'Made with <a href="https://documenter.juliadocs.org/stable/" target="_blank"><strong>Documenter.jl</strong></a>, <a href="https://vitepress.dev" target="_blank"><strong>VitePress</strong></a> and <a href="https://luxdl.github.io/DocumenterVitepress.jl/stable" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>Released under the BSD 3-Clause License. Powered by the <a href="https://www.julialang.org" target="_blank">Julia Programming Language</a>.<br>',
        }
    }
})