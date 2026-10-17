/**
 * NOVA — 3D Holographic AI Security Droid Mascot for LATIS
 * Powered by Three.js
 * 
 * Features:
 * - Floating cute stylized robotic droid with rounded porcelain/glass head, metallic visor & glowing expressive digital eyes.
 * - Interactive states:
 *     - IDLE / MONITORING: Gentle hover bobbing, cyan-teal glowing eye slit, blinking, looking around.
 *     - SCANNING: Eyes blink rapidly into blue data streams, antenna pulses.
 *     - CRITICAL THREAT: Eyes narrow into angled crimson red alert slits, head tilts, red alarm rings pulse.
 *     - ERADICATING: Golden/emerald energy laser shield, ecstatic expression, confirms threat neutralization.
 * - Dynamic speech HUD with real-time commentary on telemetry, OWASP alerts, and causal isolation.
 * - Clickable interaction: Click Nova to run live diagnostics or step the stream!
 */

class NovaMascot {
    constructor() {
        this.container = document.getElementById('novaMascotContainer');
        this.speechBubble = document.getElementById('novaSpeechText');
        this.statusBadge = document.getElementById('novaMoodBadge');
        if (!this.container) return;

        this.width = 160;
        this.height = 160;
        this.state = 'IDLE'; // IDLE, SCANNING, CRITICAL, ERADICATING, SAFE
        this.targetRotation = { x: 0, y: 0 };
        this.currentRotation = { x: 0, y: 0 };
        this.eyeEmotion = 'normal'; // normal, squint, angry, happy, blink
        this.clock = new THREE.Clock();
        this.blinkTimer = 0;
        this.isBlinking = false;

        this.initThree();
        this.buildRobot();
        this.setupEvents();
        this.animate();

        this.say("Hello, Officer! I'm Nova, your AI Security Sentinel. I'm actively monitoring all internal LLM activity.");
    }

    initThree() {
        this.scene = new THREE.Scene();

        this.camera = new THREE.PerspectiveCamera(40, this.width / this.height, 0.1, 100);
        this.camera.position.set(0, 0, 5.2);

        this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
        this.scene.add(ambientLight);

        this.keyLight = new THREE.DirectionalLight(0x38bdf8, 2.2);
        this.keyLight.position.set(3, 4, 3);
        this.scene.add(this.keyLight);

        this.rimLight = new THREE.DirectionalLight(0x818cf8, 1.5);
        this.rimLight.position.set(-3, -2, -2);
        this.scene.add(this.rimLight);

        this.pointGlow = new THREE.PointLight(0x06b6d4, 2, 10);
        this.pointGlow.position.set(0, 0.2, 1.8);
        this.scene.add(this.pointGlow);
    }

    buildRobot() {
        this.robotGroup = new THREE.Group();

        // 1. Sleek Floating Droid Body (Egg / Capsule-like)
        const bodyGeo = new THREE.SphereGeometry(0.9, 36, 36);
        bodyGeo.scale(1, 1.15, 0.95);
        this.bodyMat = new THREE.MeshPhysicalMaterial({
            color: 0x0f172a,
            roughness: 0.18,
            metalness: 0.85,
            clearcoat: 1.0,
            clearcoatRoughness: 0.1,
            reflectivity: 0.9
        });
        this.body = new THREE.Mesh(bodyGeo, this.bodyMat);
        this.robotGroup.add(this.body);

        // 2. Front Visor Screen (Glossy Dark Obsidian Glass)
        const visorGeo = new THREE.SphereGeometry(0.72, 32, 24, 0, Math.PI * 2, 0, Math.PI * 0.45);
        visorGeo.rotateX(Math.PI / 2);
        this.visorMat = new THREE.MeshPhysicalMaterial({
            color: 0x020617,
            roughness: 0.05,
            metalness: 0.95,
            transmission: 0.2,
            reflectivity: 1.0
        });
        this.visor = new THREE.Mesh(visorGeo, this.visorMat);
        this.visor.position.set(0, 0.15, 0.42);
        this.visor.scale.set(0.95, 0.75, 0.55);
        this.robotGroup.add(this.visor);

        // 3. Expressive Glowing Eyes
        this.eyesGroup = new THREE.Group();

        // Standard CylinderGeometry (universally compatible across all Three.js builds)
        const eyeGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.28, 16);
        eyeGeo.scale(1, 1, 0.4);
        this.eyeMat = new THREE.MeshBasicMaterial({
            color: 0x38bdf8,
            wireframe: false
        });

        this.leftEye = new THREE.Mesh(eyeGeo, this.eyeMat);
        this.leftEye.position.set(-0.28, 0.18, 0.82);
        this.leftEye.rotation.z = 0.05;

        this.rightEye = new THREE.Mesh(eyeGeo, this.eyeMat);
        this.rightEye.position.set(0.28, 0.18, 0.82);
        this.rightEye.rotation.z = -0.05;

        this.eyesGroup.add(this.leftEye);
        this.eyesGroup.add(this.rightEye);
        this.robotGroup.add(this.eyesGroup);

        // 4. Floating Holographic Ear-Fin Antennas
        const earGeo = new THREE.CylinderGeometry(0.18, 0.18, 0.14, 24);
        earGeo.rotateZ(Math.PI / 2);
        this.earMat = new THREE.MeshStandardMaterial({
            color: 0x1e293b,
            metalness: 0.9,
            roughness: 0.3
        });

        this.leftEar = new THREE.Mesh(earGeo, this.earMat);
        this.leftEar.position.set(-0.95, 0.15, 0);
        this.rightEar = new THREE.Mesh(earGeo, this.earMat);
        this.rightEar.position.set(0.95, 0.15, 0);

        // Ear glow rings
        const ringGeo = new THREE.TorusGeometry(0.14, 0.03, 16, 32);
        ringGeo.rotateY(Math.PI / 2);
        this.ringMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });

        this.leftEarRing = new THREE.Mesh(ringGeo, this.ringMat);
        this.leftEarRing.position.set(-1.02, 0.15, 0);
        this.rightEarRing = new THREE.Mesh(ringGeo, this.ringMat);
        this.rightEarRing.position.set(1.02, 0.15, 0);

        this.robotGroup.add(this.leftEar);
        this.robotGroup.add(this.rightEar);
        this.robotGroup.add(this.leftEarRing);
        this.robotGroup.add(this.rightEarRing);

        // 5. Floating Orbital Data Rings (Shield / Holographic Gyroscope)
        const orbitGeo = new THREE.TorusGeometry(1.35, 0.02, 16, 64);
        this.orbitMat = new THREE.MeshBasicMaterial({
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.45
        });
        this.orbitRing1 = new THREE.Mesh(orbitGeo, this.orbitMat);
        this.orbitRing1.rotation.x = Math.PI / 3;
        this.robotGroup.add(this.orbitRing1);

        const orbitGeo2 = new THREE.TorusGeometry(1.48, 0.015, 16, 64);
        this.orbitRing2 = new THREE.Mesh(orbitGeo2, this.orbitMat.clone());
        this.orbitRing2.rotation.y = Math.PI / 4;
        this.orbitRing2.material.opacity = 0.3;
        this.robotGroup.add(this.orbitRing2);

        // 6. Floating Cute Cyber-Halo / Antenna Beacon
        const beaconGeo = new THREE.SphereGeometry(0.08, 16, 16);
        this.beaconMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
        this.beacon = new THREE.Mesh(beaconGeo, this.beaconMat);
        this.beacon.position.set(0, 1.45, 0);
        this.robotGroup.add(this.beacon);

        const stemGeo = new THREE.CylinderGeometry(0.02, 0.02, 0.3, 12);
        this.stem = new THREE.Mesh(stemGeo, this.earMat);
        this.stem.position.set(0, 1.25, 0);
        this.robotGroup.add(this.stem);

        this.scene.add(this.robotGroup);
    }

    setupEvents() {
        // Look at mouse cursor gently
        window.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;

            const dx = (e.clientX - centerX) / window.innerWidth;
            const dy = (e.clientY - centerY) / window.innerHeight;

            this.targetRotation.y = dx * 1.2;
            this.targetRotation.x = dy * 0.8;
        });

        // Click on mascot to interact
        this.container.addEventListener('click', () => {
            this.onClick();
        });
    }

    onClick() {
        // Trigger cute jump and voice line
        this.robotGroup.position.y += 0.35;
        this.setEmotion('happy');

        const phrases = [
            "All telemetry streams nominal! No unauthorized token exfiltrations detected.",
            "Analyzing incoming vectors... RAG embeddings and tool sandboxes are secure.",
            "Ready to isolate rogue components via counterfactual replay at your command!",
            "Did you know? OWASP LLM01 Prompt Injections account for over 60% of LLM breaches."
        ];
        const chosen = phrases[Math.floor(Math.random() * phrases.length)];
        this.say(chosen);

        setTimeout(() => {
            if (this.state === 'IDLE') this.setEmotion('normal');
        }, 3000);
    }

    say(text) {
        if (!this.speechBubble) return;
        this.speechBubble.innerText = text;
        this.speechBubble.classList.add('pop');
        setTimeout(() => {
            this.speechBubble.classList.remove('pop');
        }, 350);
    }

    setEmotion(emotion) {
        this.eyeEmotion = emotion;
        if (!this.leftEye || !this.rightEye) return;

        if (emotion === 'normal') {
            this.leftEye.scale.set(1, 1, 1);
            this.rightEye.scale.set(1, 1, 1);
            this.leftEye.rotation.z = 0.05;
            this.rightEye.rotation.z = -0.05;
        } else if (emotion === 'squint' || emotion === 'blink') {
            this.leftEye.scale.set(1, 0.15, 1);
            this.rightEye.scale.set(1, 0.15, 1);
        } else if (emotion === 'happy') {
            this.leftEye.scale.set(1.2, 0.65, 1);
            this.rightEye.scale.set(1.2, 0.65, 1);
            this.leftEye.rotation.z = -0.2;
            this.rightEye.rotation.z = 0.2;
        } else if (emotion === 'angry' || emotion === 'alert') {
            this.leftEye.scale.set(1.1, 0.5, 1);
            this.rightEye.scale.set(1.1, 0.5, 1);
            this.leftEye.rotation.z = 0.45; // angled angry brows
            this.rightEye.rotation.z = -0.45;
        }
    }

    setColorTheme(hexColor, secondaryHex = 0xffffff) {
        this.eyeMat.color.setHex(hexColor);
        this.ringMat.color.setHex(hexColor);
        this.orbitMat.color.setHex(hexColor);
        this.beaconMat.color.setHex(hexColor);
        this.pointGlow.color.setHex(hexColor);
        this.keyLight.color.setHex(hexColor);
    }

    setMascotState(state, detail = "") {
        this.state = state;
        if (this.statusBadge) {
            this.statusBadge.innerText = state;
            this.statusBadge.className = `nova-badge ${state.toLowerCase()}`;
        }

        switch (state) {
            case 'IDLE':
                this.setColorTheme(0x38bdf8);
                this.setEmotion('normal');
                break;
            case 'SCANNING':
                this.setColorTheme(0x06b6d4);
                this.setEmotion('squint');
                this.say("Scanning incoming telemetry stream for OWASP Top 10 exploits...");
                break;
            case 'CRITICAL':
                this.setColorTheme(0xef4444);
                this.setEmotion('angry');
                this.say(`CRITICAL SECURITY BREACH DETECTED! ${detail || 'High severity threat flagged.'}`);
                break;
            case 'HIGH':
                this.setColorTheme(0xf97316);
                this.setEmotion('angry');
                this.say(`HIGH RISK ALERT: ${detail || 'Potential vector injection / tool abuse.'}`);
                break;
            case 'MEDIUM':
                this.setColorTheme(0xfbbf24);
                this.setEmotion('squint');
                this.say(`MEDIUM ANOMALY: Flagged for prompt extraction boundary verification.`);
                break;
            case 'ERADICATING':
                this.setColorTheme(0x10b981);
                this.setEmotion('happy');
                this.say("Threat neutralized! Countermeasures applied and causal baseline verified safe.");
                break;
            case 'SAFE':
                this.setColorTheme(0x10b981);
                this.setEmotion('happy');
                this.say("All clear! Zero vulnerabilities detected in this trace.");
                break;
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const elapsedTime = this.clock.getElapsedTime();

        // 1. Smooth rotation toward mouse look
        this.currentRotation.x += (this.targetRotation.x - this.currentRotation.x) * 0.08;
        this.currentRotation.y += (this.targetRotation.y - this.currentRotation.y) * 0.08;

        this.robotGroup.rotation.y = this.currentRotation.y;
        this.robotGroup.rotation.x = this.currentRotation.x;

        // 2. Idle hover bobbing (gentle float)
        const hoverOffset = Math.sin(elapsedTime * 2.5) * 0.12;
        this.robotGroup.position.y += (hoverOffset - this.robotGroup.position.y) * 0.1;

        // 3. Orbital rings rotation
        if (this.orbitRing1) this.orbitRing1.rotation.z += 0.02;
        if (this.orbitRing2) this.orbitRing2.rotation.x += 0.015;

        // 4. Beacon pulse
        if (this.beacon) {
            const scale = 1.0 + Math.sin(elapsedTime * 6) * 0.25;
            this.beacon.scale.set(scale, scale, scale);
        }

        // 5. Random organic eye blink
        this.blinkTimer += 0.016;
        if (this.blinkTimer > 3.8 && !this.isBlinking && this.eyeEmotion === 'normal') {
            this.isBlinking = true;
            this.setEmotion('blink');
            setTimeout(() => {
                this.setEmotion('normal');
                this.isBlinking = false;
                this.blinkTimer = 0;
            }, 180);
        }

        // 6. Shake if Critical Alert
        if (this.state === 'CRITICAL') {
            this.robotGroup.position.x = (Math.random() - 0.5) * 0.05;
        } else {
            this.robotGroup.position.x = 0;
        }

        this.renderer.render(this.scene, this.camera);
    }
}

// Global Nova Instance
let NOVA = null;
function initNova() {
    if (!NOVA && typeof THREE !== 'undefined') {
        const el = document.getElementById('novaMascotContainer');
        if (el) {
            NOVA = new NovaMascot();
            window.NOVA = NOVA;
        }
    }
}

if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', initNova);
} else {
    initNova();
}
