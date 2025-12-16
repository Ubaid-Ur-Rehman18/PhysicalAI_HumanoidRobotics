import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      {/* Decorative floating glow orbs */}
      <div className="hero__decoration hero__decoration--1" aria-hidden="true" />
      <div className="hero__decoration hero__decoration--2" aria-hidden="true" />

      <div className="container">
        {/* Subtitle badge */}
        <span className={styles.heroBadge}>
          ROS2 • Isaac Sim • URDF • Physics
        </span>

        {/* Main title */}
        <Heading as="h1" className={clsx('hero__title', styles.heroTitle)}>
          Physical AI &amp;<br />
          <span className={styles.heroTitleAccent}>Humanoid Robotics</span>
        </Heading>

        {/* Tagline */}
        <p className={clsx('hero__subtitle', styles.heroSubtitle)}>
          {siteConfig.tagline}
        </p>

        {/* CTA Buttons */}
        <div className={styles.buttons}>
          <Link
            className={clsx('button button--primary button--lg', styles.heroButtonPrimary)}
            to="/docs/intro">
            Get Started
          </Link>
          <Link
            className={clsx('button button--secondary button--lg', styles.heroButtonSecondary)}
            to="/docs/chapters/chapter1-ros2-foundations">
            Explore Tutorials
          </Link>
        </div>

        {/* Tech stack icons row */}
        <div className={styles.techStack}>
          <span className={styles.techItem}>ROS2 Humble</span>
          <span className={styles.techDivider}>|</span>
          <span className={styles.techItem}>NVIDIA Isaac Sim</span>
          <span className={styles.techDivider}>|</span>
          <span className={styles.techItem}>Python &amp; C++</span>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Physical AI & Humanoid Robotics"
      description="Learn to build intelligent humanoid robots with ROS2, URDF, Isaac Sim, and physics simulation. Comprehensive tutorials for robotics developers.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
