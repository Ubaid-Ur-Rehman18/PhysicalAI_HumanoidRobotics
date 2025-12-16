import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Learn, Build, and Explore the Future of Intelligent Machines',
  favicon: 'img/favicon.ico',

  // Future flags
  future: {
    v4: true,
  },

  // Site URL and base
  url: 'http://localhost:3000', // local dev URL, change when deploying
  baseUrl: '/',

  // GitHub deployment config (optional)
  organizationName: 'ubaid', // replace with your GitHub username if deploying
  projectName: 'physical-ai-humanoid', // repo name

  onBrokenLinks: 'ignore',
  onBrokenMarkdownLinks: 'warn',

  // Internationalization configuration
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
    localeConfigs: {
      en: {
        label: 'English',
        direction: 'ltr',
        htmlLang: 'en-US',
      },
      ur: {
        label: 'Urdu (اردو)',
        direction: 'rtl',
        htmlLang: 'ur-PK',
      },
    },
  },

  // KaTeX stylesheet for LaTeX math rendering
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/ubaid/physical-ai-humanoid/tree/main/',
          // Add remark and rehype plugins for math support
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/ubaid/physical-ai-humanoid/tree/main/blog/',
          // Add remark and rehype plugins for math support in blog
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Humanoid Robotics',
      logo: {
        alt: 'Humanoid Robotics Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Tutorial',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/ubaid/physical-ai-humanoid',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {label: 'Getting Started', to: '/docs/intro'},
            {label: 'ROS2 Foundations', to: '/docs/chapters/chapter1-ros2-foundations'},
            {label: 'URDF Fundamentals', to: '/docs/chapters/chapter2-urdf-fundamentals'},
          ],
        },
        {
          title: 'Community',
          items: [
            {label: 'GitHub', href: 'https://github.com/Ubaid-Ur-Rehman18'},
            {label: 'LinkedIn', href: 'https://www.linkedin.com/in/ubaid-ur-rehman-5851b22b4/'},
            {label: 'Instagram', href: 'https://www.instagram.com/smartstackstudio/'},
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'Blog', to: '/blog'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics. Developed by Ubaid Ur Rehman.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.vsDark,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
