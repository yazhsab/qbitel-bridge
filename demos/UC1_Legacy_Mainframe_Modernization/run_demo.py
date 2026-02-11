#!/usr/bin/env python3
"""
QBITEL - UC1 Legacy Mainframe Modernization Demo Runner

This script provides multiple ways to run and interact with the demo:
1. Interactive CLI mode
2. Web server mode
3. Automated demonstration mode

Usage:
    python run_demo.py              # Interactive CLI
    python run_demo.py --server     # Start web server
    python run_demo.py --auto       # Automated demo
"""

import argparse
import asyncio
import json
import sys
import time
import os
from pathlib import Path
from datetime import datetime

# Add demo directory to path
DEMO_DIR = Path(__file__).parent
sys.path.insert(0, str(DEMO_DIR / "backend"))

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def print_banner():
    """Print the demo banner."""
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   {Colors.BOLD}QBITEL Bridge - Legacy Mainframe Modernization Demo{Colors.CYAN}                           ║
║   {Colors.DIM}UC1: AI-Powered COBOL/Mainframe Modernization{Colors.CYAN}                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)

def print_section(title):
    """Print a section header."""
    print(f"\n{Colors.BLUE}{'═' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'═' * 60}{Colors.ENDC}\n")

def print_success(message):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_info(message):
    """Print an info message."""
    print(f"{Colors.CYAN}ℹ {message}{Colors.ENDC}")

def print_warning(message):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_error(message):
    """Print an error message."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def print_code(code, language="python"):
    """Print formatted code."""
    print(f"\n{Colors.DIM}```{language}{Colors.ENDC}")
    for line in code.split('\n')[:30]:
        print(f"  {line}")
    if code.count('\n') > 30:
        print(f"  {Colors.DIM}... ({code.count(chr(10)) - 30} more lines){Colors.ENDC}")
    print(f"{Colors.DIM}```{Colors.ENDC}\n")

async def run_automated_demo():
    """Run the automated demonstration."""
    from app import (
        mainframe, cobol_analyzer, protocol_analyzer,
        modernization_planner, ModernizationApproach, COBOL_DIR
    )

    print_banner()
    print_info("Running automated demonstration...")
    print_info(f"Demo directory: {DEMO_DIR}")
    time.sleep(1)

    # Step 1: System Discovery
    print_section("Step 1: Legacy System Discovery")

    systems = mainframe.get_all_systems()
    print(f"Discovered {Colors.GREEN}{len(systems)}{Colors.ENDC} legacy mainframe systems:\n")

    for sys in systems:
        status_color = Colors.GREEN if sys.status.value == "active" else Colors.YELLOW
        print(f"  {Colors.BOLD}{sys.name}{Colors.ENDC} ({sys.system_id})")
        print(f"    Platform: {sys.platform}")
        print(f"    Language: {sys.language}")
        print(f"    Lines of Code: {sys.lines_of_code:,}")
        print(f"    Age: {sys.age_years} years")
        print(f"    Status: {status_color}{sys.status.value}{Colors.ENDC}")
        print(f"    Dependencies: {', '.join(sys.dependencies)}")
        print()

    time.sleep(2)

    # Step 2: COBOL Analysis
    print_section("Step 2: COBOL Code Analysis")

    cobol_files = list(COBOL_DIR.glob("*.cbl"))
    print(f"Found {Colors.GREEN}{len(cobol_files)}{Colors.ENDC} COBOL source files:\n")

    for cobol_file in cobol_files:
        print(f"  Analyzing: {Colors.CYAN}{cobol_file.name}{Colors.ENDC}")
        program = cobol_analyzer.analyze_cobol_file(str(cobol_file))

        print(f"    Program ID: {program.program_id}")
        print(f"    Lines of Code: {program.lines_of_code}")
        print(f"    Complexity Score: {program.complexity_score}")
        print(f"    Data Divisions: {program.data_divisions}")
        print(f"    Procedure Divisions: {program.procedure_divisions}")

        # Show analysis highlights
        ws = program.analysis.get("working_storage", {})
        print(f"    Working Storage Variables: {ws.get('variable_count', 0)}")

        legacy_patterns = program.analysis.get("legacy_patterns", [])
        if legacy_patterns:
            print(f"\n    {Colors.YELLOW}Legacy Patterns Detected:{Colors.ENDC}")
            for pattern in legacy_patterns:
                severity_color = Colors.RED if pattern['severity'] == 'critical' else Colors.YELLOW
                print(f"      - [{severity_color}{pattern['severity']}{Colors.ENDC}] {pattern['pattern']}: {pattern['description']}")

        opportunities = program.analysis.get("modernization_opportunities", [])
        if opportunities:
            print(f"\n    {Colors.GREEN}Modernization Opportunities:{Colors.ENDC}")
            for opp in opportunities:
                print(f"      - {opp['area']}: {opp['current']} → {opp['modern']}")

        print()

    time.sleep(2)

    # Step 3: Protocol Analysis
    print_section("Step 3: Mainframe Protocol Analysis")

    # Simulate EBCDIC customer record
    sample_data = bytes.fromhex('d1d6c8d540e2d4c9e3c840404040404040404040f1f2f3f4f5f6f7f8f9f0c1c3c3d6e4d5e340')
    print(f"Analyzing sample mainframe data ({len(sample_data)} bytes)...")
    print(f"  Raw hex: {Colors.DIM}{sample_data.hex()[:60]}...{Colors.ENDC}\n")

    analysis = protocol_analyzer.analyze_protocol(sample_data, "IBM z/OS Customer Master")

    print(f"  Detected Encoding: {Colors.GREEN}{analysis['encoding']}{Colors.ENDC}")
    print(f"  Structure Type: {analysis['structure']['type']}")
    print(f"  Record Length: {analysis['raw_length']} bytes")

    if analysis['fields']:
        print(f"\n  Detected Fields ({len(analysis['fields'])}):")
        for field in analysis['fields'][:5]:
            print(f"    - {field['name']}: offset {field['offset']}, {field['length']} bytes ({field['type']})")

    if analysis['recommendations']:
        print(f"\n  {Colors.CYAN}Recommendations:{Colors.ENDC}")
        for rec in analysis['recommendations']:
            print(f"    - {rec['issue']}")
            print(f"      Solution: {Colors.GREEN}{rec['solution']}{Colors.ENDC}")

    time.sleep(2)

    # Step 4: Code Generation
    print_section("Step 4: Modern Code Generation")

    from app import ModernCodeGenerator

    generator = ModernCodeGenerator()

    if cobol_files:
        program = cobol_analyzer.analyze_cobol_file(str(cobol_files[0]))

        print(f"Generating modern code from: {Colors.CYAN}{cobol_files[0].name}{Colors.ENDC}\n")

        # Generate Python models
        print(f"{Colors.BOLD}Generated Python Models:{Colors.ENDC}")
        models_code = generator.generate_python_model(program.analysis)
        print_code(models_code, "python")

        # Generate FastAPI endpoints
        print(f"{Colors.BOLD}Generated FastAPI Endpoints:{Colors.ENDC}")
        api_code = generator.generate_fastapi_endpoints(program.analysis)
        print_code(api_code, "python")

        # Generate SQL Schema
        print(f"{Colors.BOLD}Generated SQL Schema:{Colors.ENDC}")
        sql_code = generator.generate_sql_schema(program.analysis)
        print_code(sql_code, "sql")

    time.sleep(2)

    # Step 5: Modernization Plan
    print_section("Step 5: Modernization Plan Generation")

    system = systems[0]  # Core Banking System
    print(f"Creating modernization plan for: {Colors.CYAN}{system.name}{Colors.ENDC}\n")

    if cobol_files:
        cobol_analysis = cobol_analyzer.analyze_cobol_file(str(cobol_files[0])).analysis
    else:
        cobol_analysis = {}

    plan = modernization_planner.create_plan(
        system=system,
        cobol_analysis=cobol_analysis,
        approach=ModernizationApproach.REFACTOR,
        target_language="python"
    )

    print(f"  Plan ID: {Colors.GREEN}{plan.plan_id}{Colors.ENDC}")
    print(f"  Approach: {plan.approach.value.upper()}")
    print(f"  Risk Level: {Colors.YELLOW}{plan.risk_level.value.upper()}{Colors.ENDC}")
    print(f"  Estimated Effort: {Colors.CYAN}{plan.estimated_effort_days} person-days{Colors.ENDC}")

    print(f"\n  {Colors.BOLD}Project Phases:{Colors.ENDC}")
    total_weeks = 0
    for phase in plan.phases:
        total_weeks += phase['duration_weeks']
        print(f"\n    Phase {phase['phase']}: {phase['name']}")
        print(f"      Duration: {phase['duration_weeks']} weeks")
        print(f"      {phase['description']}")
        print(f"      Deliverables:")
        for deliverable in phase['deliverables'][:3]:
            print(f"        - {deliverable}")

    print(f"\n  Total Duration: {Colors.GREEN}{total_weeks} weeks{Colors.ENDC}")

    time.sleep(1)

    # Summary
    print_section("Demo Summary")

    print(f"""
{Colors.GREEN}Demo completed successfully!{Colors.ENDC}

{Colors.BOLD}What was demonstrated:{Colors.ENDC}
  ✓ Discovery of {len(systems)} legacy mainframe systems
  ✓ Analysis of {len(cobol_files)} COBOL source files
  ✓ Protocol reverse engineering for mainframe data formats
  ✓ Automatic generation of modern Python/FastAPI code
  ✓ Comprehensive modernization planning

{Colors.BOLD}Key Capabilities:{Colors.ENDC}
  • AI-powered COBOL code analysis
  • Legacy pattern detection and recommendations
  • EBCDIC and mainframe protocol decoding
  • Automatic code transformation
  • Risk assessment and effort estimation

{Colors.CYAN}To explore interactively, run:{Colors.ENDC}
  python run_demo.py --server

  Then open: http://localhost:8001
""")

def run_interactive_cli():
    """Run interactive CLI mode."""
    print_banner()
    print_info("Interactive CLI mode")
    print_info("Loading demo components...")

    # Import demo components
    sys.path.insert(0, str(DEMO_DIR / "backend"))
    from app import (
        mainframe, cobol_analyzer, protocol_analyzer,
        modernization_planner, ModernizationApproach, COBOL_DIR
    )

    print_success("Demo components loaded successfully\n")

    while True:
        print(f"""
{Colors.CYAN}Available Commands:{Colors.ENDC}
  1. List legacy systems
  2. Analyze COBOL files
  3. Analyze protocol data
  4. Generate modern code
  5. Create modernization plan
  6. Run full automated demo
  0. Exit
""")
        try:
            choice = input(f"{Colors.BOLD}Enter choice (0-6): {Colors.ENDC}").strip()

            if choice == "0":
                print_info("Exiting demo...")
                break
            elif choice == "1":
                systems = mainframe.get_all_systems()
                print(f"\n{Colors.GREEN}{len(systems)} systems found:{Colors.ENDC}\n")
                for sys in systems:
                    print(f"  {sys.system_id}: {sys.name} ({sys.lines_of_code:,} LOC)")
            elif choice == "2":
                cobol_files = list(COBOL_DIR.glob("*.cbl"))
                print(f"\n{Colors.GREEN}{len(cobol_files)} COBOL files:{Colors.ENDC}\n")
                for f in cobol_files:
                    program = cobol_analyzer.analyze_cobol_file(str(f))
                    print(f"  {f.name}: {program.lines_of_code} LOC, complexity {program.complexity_score}")
            elif choice == "3":
                print("\nEnter hex data (or press Enter for sample):")
                hex_input = input().strip()
                if not hex_input:
                    hex_input = 'd1d6c8d540e2d4c9e3c840404040404040404040f1f2f3f4f5f6f7f8f9f0c1c3c3d6e4d5e340'
                data = bytes.fromhex(hex_input)
                analysis = protocol_analyzer.analyze_protocol(data)
                print(f"\nEncoding: {analysis['encoding']}")
                print(f"Structure: {analysis['structure']['type']}")
                print(f"Fields detected: {len(analysis['fields'])}")
            elif choice == "4":
                cobol_files = list(COBOL_DIR.glob("*.cbl"))
                if cobol_files:
                    from app import ModernCodeGenerator
                    program = cobol_analyzer.analyze_cobol_file(str(cobol_files[0]))
                    generator = ModernCodeGenerator()
                    print("\nGenerated Python models:")
                    print(generator.generate_python_model(program.analysis)[:500])
                else:
                    print_warning("No COBOL files found")
            elif choice == "5":
                systems = mainframe.get_all_systems()
                cobol_files = list(COBOL_DIR.glob("*.cbl"))
                cobol_analysis = cobol_analyzer.analyze_cobol_file(str(cobol_files[0])).analysis if cobol_files else {}
                plan = modernization_planner.create_plan(
                    system=systems[0],
                    cobol_analysis=cobol_analysis,
                    approach=ModernizationApproach.REFACTOR
                )
                print(f"\nPlan ID: {plan.plan_id}")
                print(f"Risk: {plan.risk_level.value}")
                print(f"Effort: {plan.estimated_effort_days} days")
                print(f"Phases: {len(plan.phases)}")
            elif choice == "6":
                asyncio.run(run_automated_demo())
            else:
                print_warning("Invalid choice")
        except KeyboardInterrupt:
            print("\n")
            print_info("Interrupted. Exiting...")
            break
        except Exception as e:
            print_error(f"Error: {e}")

def run_server():
    """Start the web server."""
    print_banner()
    print_info("Starting web server...")

    try:
        import uvicorn
        print_success("Server starting at http://localhost:8001")
        print_info("Press Ctrl+C to stop\n")

        os.chdir(DEMO_DIR / "backend")
        uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
    except ImportError:
        print_error("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="QBITEL - UC1 Legacy Mainframe Modernization Demo"
    )
    parser.add_argument(
        "--server", "-s",
        action="store_true",
        help="Start web server mode"
    )
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="Run automated demonstration"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8001,
        help="Server port (default: 8001)"
    )

    args = parser.parse_args()

    if args.server:
        run_server()
    elif args.auto:
        asyncio.run(run_automated_demo())
    else:
        run_interactive_cli()

if __name__ == "__main__":
    main()
