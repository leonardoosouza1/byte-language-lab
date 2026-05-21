// Calculator — composed by the brain from its own cached code
class Calculator {
  constructor() { this.display = '0'; this.acc = null; this.op = null; }
  somar(a, b) { return a + b; }
  dividir(a, b) { return a / b; }
  compute() {
    const b = parseFloat(this.display);
    if (this.op === '+') this.acc = this.somar(this.acc, b);
    else if (this.op === '-') this.acc = this.subtrair(this.acc, b);
    else if (this.op === '*') this.acc = this.multiplicar(this.acc, b);
    else if (this.op === '/') this.acc = this.dividir(this.acc, b);
    this.display = String(this.acc);
  }
  inputDigit(d) { this.display = this.display === '0' ? d : this.display + d; }
  setOp(o) { this.acc = parseFloat(this.display); this.op = o; this.display = '0'; }
}

// DOM wiring (pattern from brain memory):
/* function addDebounceListener(input, callback, delay) {
  input.addEventListener('input', function() {
    const timer = setTimeout(() => {
      callback();
    }, delay);
    // Limpa o timer se o evento ocorrer novamente antes do delay
    input.removeEventListener('input', arguments.callee);
     */
if (typeof document !== 'undefined') {
  const calc = new Calculator();
  const out = document.getElementById('display');
  document.querySelectorAll('[data-d]').forEach(b =>
    b.addEventListener('click', () => { calc.inputDigit(b.dataset.d); out.textContent = calc.display; }));
  document.querySelectorAll('[data-op]').forEach(b =>
    b.addEventListener('click', () => { calc.setOp(b.dataset.op); }));
  const eq = document.getElementById('eq');
  if (eq) eq.addEventListener('click', () => { calc.compute(); out.textContent = calc.display; });
}
if (typeof module !== 'undefined') module.exports = Calculator;
