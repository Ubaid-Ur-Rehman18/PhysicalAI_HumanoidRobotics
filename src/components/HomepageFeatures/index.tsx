import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  icon: ReactNode;
  description: ReactNode;
};

// Inline SVG icons for robotics themes
const ROS2Icon = () => (
  <svg
    className={styles.featureIcon}
    viewBox="0 0 64 64"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <circle cx="32" cy="32" r="28" stroke="currentColor" strokeWidth="2" fill="none" />
    <circle cx="32" cy="32" r="8" fill="currentColor" />
    <path d="M32 4V20M32 44V60M4 32H20M44 32H60" stroke="currentColor" strokeWidth="2" />
    <circle cx="32" cy="12" r="4" fill="currentColor" />
    <circle cx="32" cy="52" r="4" fill="currentColor" />
    <circle cx="12" cy="32" r="4" fill="currentColor" />
    <circle cx="52" cy="32" r="4" fill="currentColor" />
  </svg>
);

const SimulationIcon = () => (
  <svg
    className={styles.featureIcon}
    viewBox="0 0 64 64"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <rect x="8" y="8" width="48" height="36" rx="2" stroke="currentColor" strokeWidth="2" fill="none" />
    <path d="M8 16H56" stroke="currentColor" strokeWidth="2" />
    <circle cx="14" cy="12" r="2" fill="currentColor" />
    <circle cx="22" cy="12" r="2" fill="currentColor" />
    <rect x="16" y="24" width="12" height="12" stroke="currentColor" strokeWidth="2" fill="none" />
    <path d="M36 24L48 30L36 36V24Z" fill="currentColor" />
    <path d="M24 52L32 44L40 52" stroke="currentColor" strokeWidth="2" fill="none" />
    <line x1="32" y1="44" x2="32" y2="56" stroke="currentColor" strokeWidth="2" />
  </svg>
);

const URDFIcon = () => (
  <svg
    className={styles.featureIcon}
    viewBox="0 0 64 64"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    {/* Robot body */}
    <rect x="22" y="20" width="20" height="24" rx="2" stroke="currentColor" strokeWidth="2" fill="none" />
    {/* Robot head */}
    <rect x="24" y="8" width="16" height="12" rx="2" stroke="currentColor" strokeWidth="2" fill="none" />
    {/* Eyes */}
    <circle cx="29" cy="14" r="2" fill="currentColor" />
    <circle cx="35" cy="14" r="2" fill="currentColor" />
    {/* Arms */}
    <path d="M22 26H12V38H18" stroke="currentColor" strokeWidth="2" fill="none" />
    <path d="M42 26H52V38H46" stroke="currentColor" strokeWidth="2" fill="none" />
    {/* Legs */}
    <path d="M26 44V56" stroke="currentColor" strokeWidth="2" />
    <path d="M38 44V56" stroke="currentColor" strokeWidth="2" />
    {/* Feet */}
    <rect x="22" y="54" width="8" height="4" rx="1" fill="currentColor" />
    <rect x="34" y="54" width="8" height="4" rx="1" fill="currentColor" />
    {/* Antenna */}
    <line x1="32" y1="8" x2="32" y2="4" stroke="currentColor" strokeWidth="2" />
    <circle cx="32" cy="3" r="2" fill="currentColor" />
  </svg>
);

const FeatureList: FeatureItem[] = [
  {
    title: 'ROS2 Integration',
    icon: <ROS2Icon />,
    description: (
      <>
        Detailed tutorials for interfacing physical robots with the{' '}
        <strong>Robot Operating System (ROS2)</strong> for real-time control,
        sensor data processing, and inter-process communication.
      </>
    ),
  },
  {
    title: 'Physics Simulation',
    icon: <SimulationIcon />,
    description: (
      <>
        Explore high-fidelity simulation with <strong>NVIDIA Isaac Sim</strong>{' '}
        for reinforcement learning, synthetic data generation, and testing
        robot behaviors in realistic virtual environments.
      </>
    ),
  },
  {
    title: 'URDF & Robot Modeling',
    icon: <URDFIcon />,
    description: (
      <>
        Learn to define robot structures using <strong>URDF</strong> (Unified
        Robot Description Format) including joints, links, visual meshes, and
        collision geometries for accurate simulation.
      </>
    ),
  },
];

function Feature({title, icon, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className={clsx('text--center', styles.featureIconWrapper)}>
        {icon}
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
        <p className={styles.featureDescription}>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2" className={styles.sectionTitle}>
            What You'll Learn
          </Heading>
          <p className={styles.sectionSubtitle}>
            Master the essential skills for building intelligent humanoid robots
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
