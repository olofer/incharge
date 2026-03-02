/*

Source trajectories are defined with piecewise cubic B-splines.
The smooth paths have continuous acceleration, but do not pass through its control points.

*/

function bcubic_coefs(f) {
  const fn1 = f[0];
  const f0 = f[1];
  const f1 = f[2];
  const f2 = f[3];
  c0 = (fn1 + 4.0 * f0 + f1) / 6.0;
  c1 = (-3.0 * fn1 + 3.0 * f1) / 6.0;
  c2 = (3.0 * fn1 - 6.0 * f0 + 3.0 * f1) / 6.0;
  c3 = (-fn1 + 3.0 * f0 - 3.0 * f1 + f2) / 6.0;
  return [c0, c1, c2, c3];
}

function spline_value(c, t) {
  const t2 = t * t;
  return c[0] + c[1] * t + c[2] * t2 + c[3] * t2 * t;
}

function spline_derivative(c, t) {
  return c[1] + 2.0 * c[2] * t + 3.0 * c[3] * t * t;
}

function cubic_interp(theta, omega, vertices) {
  const npts = vertices.length / 2;
  const delta_theta = 2.0 * Math.PI / npts;
  const dscale = omega / delta_theta;
  while (theta < -Math.PI) theta += 2 * Math.PI;
  while (theta > Math.PI) theta -= 2 * Math.PI;
  const z = (theta + Math.PI) / delta_theta;
  const idx = Math.floor(z);
  const w = z - idx;
  const in1 = (idx + npts - 1) % npts;
  const i0 = idx % npts;
  const i1 = (idx + 1) % npts;
  const i2 = (idx + 2) % npts;
  const cx = bcubic_coefs([vertices[2 * in1 + 0], vertices[2 * i0 + 0], vertices[2 * i1 + 0], vertices[2 * i2 + 0]]);
  const cy = bcubic_coefs([vertices[2 * in1 + 1], vertices[2 * i0 + 1], vertices[2 * i1 + 1], vertices[2 * i2 + 1]]);
  const rx = spline_value(cx, w);
  const ry = spline_value(cy, w);
  const vx = spline_derivative(cx, w) * dscale;
  const vy = spline_derivative(cy, w) * dscale;
  return [rx, ry, vx, vy];
}

function find_maximum_velocity(omega, vertices, overfactor) {
  const npts = vertices.length / 2;
  const nsamples = overfactor * npts; // oversample the path
  let max_vsq = 0.0;
  for (let i = 0; i < nsamples; i++) {
    const thetai = i * 2 * Math.PI / nsamples;
    const rv = cubic_interp(thetai, omega, vertices);
    const vsq = rv[2] * rv[2] + rv[3] * rv[3];
    if (vsq > max_vsq) max_vsq = vsq;
  }
  return Math.sqrt(max_vsq);
}

function canvas_coord(xy, wh, z) {
  const aspect = wh[0] / wh[1];
  const hw = wh[0] / 2;
  const hh = wh[1] / 2;
  const x = hw + xy[0] * hw / aspect / z;
  const y = hh - xy[1] * hh / z;
  return [x, y];
}

function draw_path_on_canvas(ctx, wh, zoomlvl, vertices, overfactor) {
  const npts = vertices.length / 2;
  const nsamples = overfactor * npts;
  const delta_theta = 2 * Math.PI / nsamples;
  ctx.beginPath();
  let xy_ = cubic_interp(-Math.PI + 0 * delta_theta, 1.0, vertices);
  let coords = canvas_coord([xy_[0], xy_[1]], wh, zoomlvl);
  ctx.moveTo(coords[0], coords[1]);
  for (let i = 1; i < nsamples; i++) {
    xy_ = cubic_interp(-Math.PI + i * delta_theta, 1.0, vertices);
    coords = canvas_coord([xy_[0], xy_[1]], wh, zoomlvl);
    ctx.lineTo(coords[0], coords[1])
  }
  xy_ = cubic_interp(-Math.PI + nsamples * delta_theta, 1.0, vertices);
  coords = canvas_coord([xy_[0], xy_[1]], wh, zoomlvl);
  ctx.lineTo(coords[0], coords[1]);
  ctx.stroke();
}

const numSourceVertices = 32; // need to match the number in the shader code
const sourceVertices = new Float32Array(2 * numSourceVertices);

function preset_path_0(theta) {
  return [Math.cos(theta), Math.sin(theta)];
}

function preset_path_1(theta) {
  const shift = 0.01;
  return [Math.sign(Math.cos(theta + shift)), Math.sign(Math.sin(theta + shift))];
}

function preset_path_2(theta) {
  const xi = theta / (2 * Math.PI);
  return [xi, xi];
}

function preset_path_3(theta) {
  return [Math.cos(2.0 * theta), Math.sin(3.0 * theta)];
}

function preset_path_4(theta) {
  const r = 2.0 + 0.25 * Math.cos(10 * theta);
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

function preset_path_5(theta) {
  const r = 2.0;
  const phase = 0.10 * Math.cos(10 * theta)
  return [r * Math.cos(theta + phase), r * Math.sin(theta + phase)];
}

function preset_path_6(theta) {
  const cs = Math.cos(theta);
  const sn = Math.sin(theta);
  return [cs * Math.sqrt(Math.abs(cs)), sn * Math.sqrt(Math.abs(sn))];
}

function preset_path_7(theta) {
  return [3.0 * Math.sin(theta), 0.0];
}

const PATH_FUNCTIONS = [
  preset_path_0,
  preset_path_1,
  preset_path_2,
  preset_path_3,
  preset_path_4,
  preset_path_5,
  preset_path_6,
  preset_path_7
];

function generate_preset_path(idx) {
  const delta_theta = 2.0 * Math.PI / numSourceVertices;
  const fxy = PATH_FUNCTIONS[idx];
  for (let i = 0; i < numSourceVertices; i++) {
    k = 2 * i;
    let thetai = -Math.PI + i * delta_theta;
    let xy = fxy(thetai);
    sourceVertices[k + 0] = xy[0];
    sourceVertices[k + 1] = xy[1];
  }
}

const SCANFACTOR = 8;
let IPATH = 1;
generate_preset_path(IPATH);
let VMAX = find_maximum_velocity(1.0, sourceVertices, SCANFACTOR);
console.log(VMAX);

// Get canvas and context
const canvasgl = document.getElementById('gl-canvas');
const gl = canvasgl.getContext('webgl2');
if (!gl) {
  alert('WebGL2 not supported');
  throw new Error('WebGL2 not supported');
}

// The standard "canvas" is for text overlay
const canvas2d = document.getElementById('2d-canvas');

// Set canvas to fullscreen
function resizeCanvas() {
  canvasgl.width = window.innerWidth;
  canvasgl.height = window.innerHeight;
  gl.viewport(0, 0, canvasgl.width, canvasgl.height);

  canvas2d.width = window.innerWidth;
  canvas2d.height = window.innerHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(gl, vertexShader, fragmentShader) {
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(program));
    return null;
  }
  return program;
}

const vertexShaderSource = document.getElementById('vertex-shader').textContent.trim();
const fragmentShaderSource = document.getElementById('fragment-shader').textContent.trim();

const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
const program = createProgram(gl, vertexShader, fragmentShader);
if (!program) throw new Error('Program creation failed');

// Set up attributes and uniforms
const positionAttributeLocation = gl.getAttribLocation(program, 'a_position');
const resolutionUniformLocation = gl.getUniformLocation(program, 'u_resolution');
const sourceUniformLocation = gl.getUniformLocation(program, 'u_source_vertices');
const timeUniformLocation = gl.getUniformLocation(program, 'u_time');
const zoomUniformLocation = gl.getUniformLocation(program, 'u_zoom');
const contrastUniformLocation = gl.getUniformLocation(program, 'u_contrast');
const vmaxUniformLocation = gl.getUniformLocation(program, 'u_vmax');
const betaUniformLocation = gl.getUniformLocation(program, 'u_beta');
const freqUniformLocation = gl.getUniformLocation(program, 'u_freq');
const styleUniformLocation = gl.getUniformLocation(program, 'u_style');
const dipoleUniformLocation = gl.getUniformLocation(program, 'u_dipole');

// Create buffer for a full-screen quad
const positionBuffer = gl.createBuffer();
const positions = new Float32Array([
  -1, -1,
  1, -1,
  -1, 1,
  -1, 1,
  1, -1,
  1, 1
]);
gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

// Set up VAO
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
gl.enableVertexAttribArray(positionAttributeLocation);
gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

// Animation loop
const numPlotStyles = 5;
const styleName = ["dens. |B|^2", "dens. |E|^2", "pot. |A|^2", "pot. |Phi|", "flow |S|"];
let plotStyle = 3;
let dipoleToggle = 0;
const betaFPSfilter = 1.0 / 100.0;
let filteredFPS = 0.0;
let lastTime = performance.now();
let simTime = 0.0;
let zoomLevel = 8.0;
let betaLevel = 0.85;
let freqValue = 0.20;
let contrastLevel = 1.0;
let showPath = false;
let showText = true;

function keyDownEvent(e) {
  let code = e.keyCode;
  let key = e.key;

  if (key == 'Tab') {
    plotStyle += e.shiftKey ? -1 : 1;
    if (plotStyle < 0) plotStyle = numPlotStyles - 1;
    if (plotStyle == numPlotStyles) plotStyle = 0;
    e.preventDefault();
    return;
  }

  if (key == ' ') {
    showPath = !showPath;
    e.preventDefault();
    return;
  }

  if ((key == 'b' || key == 'B')) {
    if (e.shiftKey) {
      // dial beta down
      betaLevel *= 0.95;
      if (betaLevel < 0.005) betaLevel = 0.005;
      return;
    }
    // crank beta up
    betaLevel /= 0.95;
    if (betaLevel > 0.995) betaLevel = 0.995;
    return;
  }

  if (key == 'f' || key == 'F') {
    freqValue += e.shiftKey ? -0.05 : 0.05;
    if (freqValue < 0.05) freqValue = 0.05;
    return;
  }

  if (code == 38 && e.shiftKey) { // shift + up
    zoomLevel *= 0.80;
    return;
  }

  if (code == 40 && e.shiftKey) { // shift + down
    zoomLevel *= 1.25;
    return;
  }

  if (code == 38 || code == 40) { // up/down keys cycles through the preset paths
    if (code == 38) IPATH = (IPATH + 1) % PATH_FUNCTIONS.length;
    if (code == 40) IPATH = (IPATH == 0 ? PATH_FUNCTIONS.length - 1 : IPATH - 1);
    generate_preset_path(IPATH);
    VMAX = find_maximum_velocity(1.0, sourceVertices, SCANFACTOR);
    console.log([IPATH, VMAX]);
    return;
  }

  if (key == 'n' || key == 'N') {
    // Add a little bit off random noise to the current path
    for (let i = 0; i < numSourceVertices; i++) {
      sourceVertices[2 * i + 0] += (2 * Math.random() - 1) * 0.01;
      sourceVertices[2 * i + 1] += (2 * Math.random() - 1) * 0.01;
    }
    VMAX = find_maximum_velocity(1.0, sourceVertices, SCANFACTOR);
    console.log(["added noise", VMAX]);
    return;
  }

  if (key == 'r' || key == 'R') {
    simTime = 0.0;
    contrastLevel = 1.0;
    zoomLevel = 8.0;
    plotStyle = 3;
    contrastLevel = 1.0;
    IPATH = 1;
    generate_preset_path(IPATH);
    VMAX = find_maximum_velocity(1.0, sourceVertices, SCANFACTOR);
    return;
  }

  if (key == 'c' || key == 'C') {
    contrastLevel *= (e.shiftKey ? 1.25 : 0.80);
    console.log("contrast", contrastLevel);
    return;
  }

  if (key == 'h' || key == 'H') {
    showText = !showText;
    return;
  }

  if (key == 'd' || key == 'D') {
    dipoleToggle = (dipoleToggle + 1) % 2;
    console.log("dipoleToggle", dipoleToggle);
    return;
  }

}

window.addEventListener('keydown', keyDownEvent);

function render() {
  const time = performance.now();
  const elapsedTimeSeconds = (time - lastTime) / 1000; // Time in seconds
  lastTime = time;
  simTime += elapsedTimeSeconds;

  if (elapsedTimeSeconds > 0.0 && elapsedTimeSeconds < 1.0)
    filteredFPS = (betaFPSfilter) * (1.0 / elapsedTimeSeconds) + (1.0 - betaFPSfilter) * filteredFPS;

  // Update uniforms
  gl.useProgram(program);
  gl.uniform2f(resolutionUniformLocation, canvasgl.width, canvasgl.height);
  gl.uniform2fv(sourceUniformLocation, sourceVertices);
  gl.uniform1f(timeUniformLocation, simTime);
  gl.uniform1f(zoomUniformLocation, zoomLevel);
  gl.uniform1f(contrastUniformLocation, contrastLevel);
  gl.uniform1f(betaUniformLocation, betaLevel);
  gl.uniform1f(vmaxUniformLocation, VMAX);
  gl.uniform1f(freqUniformLocation, freqValue);
  gl.uniform1i(styleUniformLocation, plotStyle);
  gl.uniform1i(dipoleUniformLocation, dipoleToggle);

  // Clear and draw
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 6);

  // Text overlay
  const ctx = canvas2d.getContext('2d');
  //ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas2d.width, canvas2d.height);
  ctx.globalAlpha = 1.0;

  if (showText) {
    ctx.fillStyle = 'white';
    ctx.font = '20px Arial';
    ctx.fillText('<fps> = ' + filteredFPS.toFixed(1), 20.0, canvas2d.height - 25.0);
    ctx.fillText('[tab] ' + styleName[plotStyle] + ', [b] beta: ' + betaLevel.toFixed(4) + ', [f] (anim.) freq: ' + freqValue.toFixed(4), 20.0, 25.0);
    ctx.fillText('[up/dn] path preset #' + IPATH.toFixed(0) + ' [space] to show [c] edit contrast [d] toggle dipole', 20.0, 45.0);
    ctx.fillText('[shift+up/dn] zoom in/out' + ' [h] hide/show text [r] reset', 20.0, 65.0);
  }

  if (showPath) {
    ctx.lineWidth = 4;
    ctx.strokeStyle = 'rgba(255,255,255,0.33)';
    const wh = [canvas2d.width, canvas2d.height];
    draw_path_on_canvas(ctx, wh, zoomLevel, sourceVertices, 5);
  }

  requestAnimationFrame(render);
}

// Start animation
gl.useProgram(program);
render();
