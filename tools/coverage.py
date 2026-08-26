#!/usr/bin/env python3
# Copyright 2025 Irreducible Inc.
"""
Rust Code Coverage Tool

A Python-based coverage tool that provides clean, formatted output for cargo-llvm-cov.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


class CoverageRunner:
    def __init__(self):
        self.workspace_root = self._get_workspace_root()
        self.workspace_name = os.path.basename(self.workspace_root)
        self._regression_detected = False
        # ANSI color codes
        self.colors = {
            'green': '\033[92m',
            'yellow': '\033[38;5;220m',  # Bright gold instead of dull yellow
            'red': '\033[91m',
            'reset': '\033[0m',
            'bold': '\033[1m'
        }

    def _get_workspace_root(self) -> str:
        """Get the workspace root directory from cargo metadata."""
        try:
            result = subprocess.run(
                ["cargo", "metadata", "--no-deps", "--format-version", "1"],
                capture_output=True,
                text=True,
                check=True
            )
            metadata = json.loads(result.stdout)
            return metadata["workspace_root"]
        except:
            return os.getcwd()

    def _get_all_packages(self) -> List[str]:
        """Get all packages in the workspace."""
        try:
            result = subprocess.run(
                ["cargo", "metadata", "--no-deps", "--format-version", "1"],
                capture_output=True,
                text=True,
                check=True
            )
            metadata = json.loads(result.stdout)
            return [pkg["name"] for pkg in metadata["packages"]]
        except:
            return []

    def _check_cargo_llvm_cov(self):
        """Check if cargo-llvm-cov is installed."""
        try:
            subprocess.run(["cargo", "llvm-cov", "--version"],
                         capture_output=True, check=True)
        except:
            print("ERROR: cargo-llvm-cov not found. Installing...")
            subprocess.run(["cargo", "install", "cargo-llvm-cov"], check=True)

    def _build_command(self, args) -> List[str]:
        """Build the cargo llvm-cov command."""
        cmd = ["cargo", "llvm-cov", "--all-features", "--ignore-run-fail"]

        if args.workspace:
            cmd.append("--workspace")

        for pkg in args.packages:
            cmd.extend(["-p", pkg])

        for exclude in args.exclude:
            cmd.extend(["--exclude", exclude])

        if args.lib:
            cmd.append("--lib")

        for test in args.test:
            cmd.extend(["--test", test])

        if not args.debug:
            cmd.append("--release")

        if args.no_cfg_coverage_nightly:
            cmd.append("--no-cfg-coverage-nightly")

        # Add output format
        if args.format == "html":
            cmd.append("--html")
        elif args.format == "json":
            cmd.extend(["--json", "--output-path", "coverage.json"])
        elif args.format == "lcov":
            cmd.extend(["--lcov", "--output-path", "lcov.info"])

        return cmd

    def _format_path(self, path: str) -> str:
        """Format file path to be relative to workspace."""
        # Remove everything up to and including the workspace name
        pattern = f".*{self.workspace_name}/"
        return re.sub(pattern, "", path)

    def _colorize_percentage(self, percentage: str) -> str:
        """Add color to percentage based on value."""
        try:
            # Remove % sign and convert to float
            value = float(percentage.rstrip('%'))
            if value >= 95:
                # Excellent coverage - green + bold
                return f"{self.colors['bold']}{self.colors['green']}{percentage}{self.colors['reset']}"
            elif value >= 80:
                # Good coverage - green
                return f"{self.colors['green']}{percentage}{self.colors['reset']}"
            elif value >= 50:
                # Moderate coverage - yellow
                return f"{self.colors['yellow']}{percentage}{self.colors['reset']}"
            else:
                # Poor coverage - red
                return f"{self.colors['red']}{percentage}{self.colors['reset']}"
        except:
            return percentage

    def _parse_coverage_line(self, line: str) -> Optional[Dict]:
        """Parse a coverage report line."""
        # Skip header and separator lines
        if line.startswith("Filename") or line.startswith("-") or not line.strip():
            return None

        # Skip .cargo and .rustup files
        if ".cargo/" in line or ".rustup/" in line:
            return None

        # Skip TOTAL line
        if line.strip().startswith("TOTAL"):
            return None

        # Parse the coverage data using more flexible approach
        # The format has many columns and spaces, so we need to be careful
        parts = line.split()
        if len(parts) < 13:
            return None

        try:
            # Extract filename (first part)
            filename = parts[0]

            # Find numeric columns - they follow a pattern
            # regions, missed_regions, cover%, functions, missed_functions, executed%, lines, missed_lines, cover%, branches, missed_branches, cover%
            return {
                "filename": self._format_path(filename),
                "regions": {
                    "total": int(parts[1]),
                    "missed": int(parts[2]),
                    "coverage": parts[3]
                },
                "functions": {
                    "total": int(parts[4]),
                    "missed": int(parts[5]),
                    "coverage": parts[6]
                },
                "lines": {
                    "total": int(parts[7]),
                    "missed": int(parts[8]),
                    "coverage": parts[9]
                }
            }
        except (ValueError, IndexError):
            return None

    def _filter_output(self, output: str, package_filter: Optional[List[str]] = None) -> Tuple[str, List[Dict]]:
        """Filter and format the coverage output."""
        lines = output.split('\n')
        formatted_lines = []
        coverage_data = []
        table_data = []
        in_table = False

        for line in lines:
            # Skip info messages
            if "info: cargo-llvm-cov currently setting cfg" in line:
                continue

            # Detect start of coverage table
            if line.startswith("Filename") and "Regions" in line:
                in_table = True
                continue

            if in_table:
                if line.startswith("-"):
                    continue

                if line.strip() == "" or "warning:" in line:
                    # End of table - format and output it
                    if table_data and HAS_TABULATE:
                        # Add header
                        formatted_lines.append("")
                        formatted_lines.append(self._format_coverage_table(table_data))
                        table_data = []
                    in_table = False
                    if line.strip() == "":
                        formatted_lines.append("")
                    else:
                        formatted_lines.append(line)  # Keep warning line
                    continue

                # Skip .cargo and .rustup lines
                if ".cargo/" in line or ".rustup/" in line:
                    continue

                # For TOTAL line, skip it since we'll show our own summary
                if line.strip().startswith("TOTAL"):
                    continue

                # Process coverage data lines
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 13:
                        # Parse the line
                        filename = parts[0]

                        # Apply filters
                        if package_filter:
                            # When testing a specific package, files without crate prefix belong to that package
                            # e.g., "circuits/base64.rs" belongs to the package being tested
                            # while "field/src/lib.rs" is from another crate
                            if not any(filter_path in filename for filter_path in package_filter):
                                # Check if this is a local file (no crate prefix)
                                has_crate_prefix = any(prefix in filename for prefix in
                                                     ['field/', 'frontend/', 'prover/', 'transcript/',
                                                      'utils/', 'verifier/'])
                                if has_crate_prefix:
                                    continue  # Skip files from other crates

                        # Format the path
                        formatted_path = self._format_path(filename)

                        # Extract coverage data
                        try:
                            row_data = {
                                'filename': formatted_path,
                                'lines': {
                                    'total': int(parts[7]),
                                    'missed': int(parts[8]),
                                    'percent': parts[9]
                                },
                                'functions': {
                                    'total': int(parts[4]),
                                    'missed': int(parts[5]),
                                    'percent': parts[6]
                                }
                            }
                            table_data.append(row_data)
                            coverage_data.append(row_data)
                        except:
                            pass
            else:
                # Keep all other output (test results, etc.)
                # Format test output for better readability
                if "test result:" in line:
                    formatted_lines.append(f"\n{line}")
                elif "warning:" in line:
                    formatted_lines.append(f"\nWARNING: {line}")
                elif line.strip().startswith("Running"):
                    formatted_lines.append(f"\n{line.strip()}")
                else:
                    formatted_lines.append(line)

        # Output any remaining table data
        if table_data and HAS_TABULATE:
            formatted_lines.append("")
            formatted_lines.append(self._format_coverage_table(table_data))

        return '\n'.join(formatted_lines), coverage_data

    def _format_coverage_table(self, table_data: List[Dict]) -> str:
        """Format coverage data as a nice table with colors."""
        if not table_data:
            return ""

        # Add section header
        output = ["File Coverage Details"]

        headers = ['File', 'Lines', 'Coverage', 'Functions', 'Coverage']
        rows = []

        for item in table_data:
            # Calculate covered lines/functions
            lines_covered = item['lines']['total'] - item['lines']['missed']
            funcs_covered = item['functions']['total'] - item['functions']['missed']

            # Format filename with proper padding
            filename = item['filename']
            if len(filename) > 45:
                filename = "..." + filename[-42:]  # Truncate long paths

            row = [
                filename,
                f"{lines_covered:>3}/{item['lines']['total']:<3}",
                self._colorize_percentage(item['lines']['percent']),
                f"{funcs_covered:>2}/{item['functions']['total']:<2}",
                self._colorize_percentage(item['functions']['percent'])
            ]
            rows.append(row)

        table = tabulate(rows, headers=headers, tablefmt='simple', colalign=('left', 'center', 'center', 'center', 'center'))
        output.append(table)

        return '\n'.join(output)

    def _format_total_line(self, parts: List[str]) -> str:
        """Format the TOTAL line with colors."""
        if HAS_TABULATE:
            # Extract percentages
            try:
                line_pct = parts[9]
                func_pct = parts[6]

                total_str = f"\n{self.colors['bold']}TOTAL{self.colors['reset']}"
                line_str = f"Lines: {self._colorize_percentage(line_pct)}"
                func_str = f"Functions: {self._colorize_percentage(func_pct)}"

                return f"{total_str:<30} {line_str:<30} {func_str}"
            except:
                return " ".join(parts)
        else:
            return " ".join(parts)

    def _calculate_summary(self, coverage_data: List[Dict]) -> Dict[str, float]:
        """Calculate coverage summary from parsed data."""
        total_lines = sum(item["lines"]["total"] for item in coverage_data)
        covered_lines = sum(item["lines"]["total"] - item["lines"]["missed"] for item in coverage_data)

        total_functions = sum(item["functions"]["total"] for item in coverage_data)
        covered_functions = sum(item["functions"]["total"] - item["functions"]["missed"] for item in coverage_data)

        line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        function_coverage = (covered_functions / total_functions * 100) if total_functions > 0 else 0

        return {
            "line_coverage": line_coverage,
            "function_coverage": function_coverage,
            "total_lines": total_lines,
            "covered_lines": covered_lines,
            "total_functions": total_functions,
            "covered_functions": covered_functions
        }

    def run(self, args):
        """Run coverage analysis."""
        self._check_cargo_llvm_cov()

        print("\nRust Code Coverage Analysis")

        # Determine what packages to run
        if args.all_packages:
            packages = self._get_all_packages()
            print(f"Found packages: {', '.join(packages)}")
        elif args.packages:
            packages = args.packages
        else:
            packages = []

        # Run coverage and collect stats
        stats = None
        if len(packages) > 1:
            # Multiple packages - run with aggregate
            stats = self._run_multiple_packages_with_aggregate_stats(packages, args)
        elif len(packages) == 1:
            # Single package
            args_copy = argparse.Namespace(**vars(args))
            args_copy.packages = packages
            stats = self._run_normal(args_copy, return_stats=True)
            # Need to wrap single package stats for consistency
            if stats:
                stats['package_stats'] = [{
                    'name': packages[0],
                    'line_pct': (stats['covered_lines'] / stats['total_lines'] * 100) if stats['total_lines'] > 0 else 0,
                    'function_pct': (stats['covered_functions'] / stats['total_functions'] * 100) if stats['total_functions'] > 0 else 0,
                    'covered_lines': stats['covered_lines'],
                    'total_lines': stats['total_lines'],
                    'covered_functions': stats['covered_functions'],
                    'total_functions': stats['total_functions']
                }]
                # Also set all_coverage_data for file details
                stats['all_coverage_data'] = stats.get('coverage_data', [])
        else:
            # No specific packages - run workspace
            self._run_normal(args)
            
        # Handle baseline operations
        if args.save_baseline and stats:
            self._save_baseline(args.save_baseline, stats)
            
        if args.compare_baseline and stats:
            baseline = self._load_baseline(args.compare_baseline)
            passed = self._compare_with_baseline(baseline, stats)
            
            if args.fail_on_regression and not passed:
                sys.exit(1)

    def _run_normal(self, args, return_stats=False):
        """Run coverage in normal mode."""
        # Build package filter
        package_filter = []
        if args.packages:
            for pkg in args.packages:
                # Remove binius- prefix for crate path
                crate_name = pkg[7:] if pkg.startswith("binius-") else pkg
                # The paths in coverage output don't have 'crates/' prefix
                package_filter.append(f"{crate_name}/src/")

        # Clean previous coverage data
        clean_cmd = ["cargo", "llvm-cov", "clean"]
        if args.workspace:
            clean_cmd.append("--workspace")
        for pkg in args.packages:
            clean_cmd.extend(["-p", pkg])
        subprocess.run(clean_cmd, capture_output=True)

        if args.packages:
            pkg_names = ", ".join(args.packages)
            print(f"\nTarget package{'s' if len(args.packages) > 1 else ''}: {pkg_names}")
        sys.stdout.flush()

        # Run coverage
        cmd = self._build_command(args)
        env = os.environ.copy()
        env["CARGO_TERM_COLOR"] = "always"

        if args.format in ["html", "json", "lcov"]:
            # For file outputs, just run the command
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            # Filter info message from stderr
            stderr = '\n'.join(line for line in result.stderr.split('\n')
                             if "info: cargo-llvm-cov currently setting cfg" not in line)
            if stderr:
                print(stderr, file=sys.stderr)

            if args.format == "html":
                print("\nHTML coverage report generated")
                print("   Location: target/llvm-cov/html/")
                print("   View: target/llvm-cov/html/index.html")
                if args.open:
                    import platform
                    if platform.system() == "Darwin":
                        subprocess.run(["open", "target/llvm-cov/html/index.html"])
                    elif platform.system() == "Linux":
                        subprocess.run(["xdg-open", "target/llvm-cov/html/index.html"])
            elif args.format == "json":
                # Also generate summary
                subprocess.run(["cargo", "llvm-cov", "report", "--json",
                              "--output-path", "coverage-summary.json"] +
                              (["--release"] if not args.debug else []),
                              capture_output=True)
                print("\nJSON coverage reports generated")
                print("   Detailed: coverage.json")
                print("   Summary: coverage-summary.json")
            elif args.format == "lcov":
                print("\nLCOV report generated")
                print("   File: lcov.info")
                print("   Use with IDE coverage viewers or CI services")
        else:
            # Summary format - show live output during test run
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as temp_out:
                temp_output_path = temp_out.name

            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_json:
                temp_json_path = temp_json.name

            try:
                # Run coverage with threaded progress indicator
                process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)

                # Shared state for thread communication
                progress_state = {
                    'running': True,
                    'current_test': 'Initializing (may take a while)... ',
                    'tests_completed': 0,
                    'coverage_started': False
                }

                # Spinner thread function
                def spinner_thread():
                    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                    spinner_idx = 0

                    while progress_state['running']:
                        if progress_state['tests_completed'] > 0:
                            status = f"{spinner[spinner_idx]} Running tests... [{progress_state['tests_completed']} tests] {progress_state['current_test'][:50]}"
                        else:
                            status = f"{spinner[spinner_idx]} {progress_state['current_test'][:70]}"
                        # Clear to end of line using ANSI escape code
                        print(f"\r{status}\033[K", end='', flush=True)
                        spinner_idx = (spinner_idx + 1) % len(spinner)
                        time.sleep(0.1)

                    # Clear the line when done
                    print("\r" + " " * 80 + "\r", end='', flush=True)

                # Start spinner thread
                spinner_t = threading.Thread(target=spinner_thread)
                spinner_t.start()

                # Process output
                full_output_lines = []

                for line in process.stdout:
                    full_output_lines.append(line)

                    # Detect coverage table start
                    if "Filename" in line and "Regions" in line:
                        progress_state['coverage_started'] = True
                        progress_state['running'] = False
                        break

                    # Track test progress
                    if not progress_state['coverage_started']:
                        # Detect running tests
                        if "Running" in line and ("target/" in line or "unittests" in line):
                            # Extract test binary name
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if "target/" in part:
                                    test_name = part.split('/')[-1].strip('()')
                                    progress_state['current_test'] = f"Running {test_name}"
                                    break
                        elif "test " in line and " ... " in line:
                            # Individual test running
                            test_match = re.search(r'test (\S+)', line)
                            if test_match:
                                progress_state['current_test'] = f"Test: {test_match.group(1)}"
                        elif (" passed" in line or " failed" in line) and "test" in line.lower():
                            # Test completed - count occurrences of "passed" or "failed"
                            # Some lines might have multiple test results
                            if " passed" in line:
                                progress_state['tests_completed'] += line.count(" passed")
                            if " failed" in line:
                                progress_state['tests_completed'] += line.count(" failed")
                        elif "test result:" in line:
                            # Test suite completed
                            progress_state['current_test'] = "Generating coverage report..."
                        elif "Compiling" in line:
                            # Compilation phase
                            progress_state['current_test'] = "Compiling..."
                        elif "Finished" in line and ("dev" in line or "release" in line):
                            # Compilation finished
                            progress_state['current_test'] = "Starting tests..."

                # Continue reading rest of output
                for line in process.stdout:
                    full_output_lines.append(line)

                process.wait()
                full_output = ''.join(full_output_lines)

                # Stop spinner
                progress_state['running'] = False
                spinner_t.join()

                # Save for processing
                with open(temp_output_path, 'w') as outfile:
                    outfile.write(full_output)

                # Now process and show the formatted coverage table
                formatted_output, coverage_data = self._filter_output(full_output, package_filter)

                # Always print the coverage table part (not test output again)
                output_lines = formatted_output.split('\n')
                in_coverage = False
                for line in output_lines:
                    if "File Coverage Details" in line:
                        in_coverage = True
                    if in_coverage:
                        print(line)

                # Generate JSON report for accurate summary
                report_cmd = ["cargo", "llvm-cov", "report", "--json",
                             "--output-path", temp_json_path]
                if not args.debug:
                    report_cmd.append("--release")
                if args.packages:
                    for pkg in args.packages:
                        report_cmd.extend(["-p", pkg])
                elif args.workspace:
                    report_cmd.append("--workspace")

                subprocess.run(report_cmd, capture_output=True)

                # Calculate summary
                try:
                    with open(temp_json_path, 'r') as f:
                        data = json.load(f)

                    if package_filter and args.packages:
                        # Filter JSON data for specific packages
                        files = data['data'][0]['files']
                        # Also check with 'crates/' prefix since JSON might have different paths
                        extended_filter = package_filter + [f"crates/{pf}" for pf in package_filter]
                        filtered_files = [f for f in files
                                        if any(filter_path in f['filename'] for filter_path in extended_filter)]

                        if filtered_files:
                            total_lines = sum(f['summary']['lines']['count'] for f in filtered_files)
                            covered_lines = sum(f['summary']['lines']['covered'] for f in filtered_files)
                            total_functions = sum(f['summary']['functions']['count'] for f in filtered_files)
                            covered_functions = sum(f['summary']['functions']['covered'] for f in filtered_files)

                            line_pct = (covered_lines / total_lines * 100) if total_lines > 0 else 0
                            function_pct = (covered_functions / total_functions * 100) if total_functions > 0 else 0
                        else:
                            line_pct = function_pct = 0
                    else:
                        # Use overall totals
                        totals = data['data'][0]['totals']
                        line_pct = totals['lines']['percent']
                        function_pct = totals['functions']['percent']

                    # Always print the summary
                    print("\nCoverage Summary:")
                    print(f"   Line Coverage:     {self._colorize_percentage(f'{line_pct:.2f}%')}")
                    print(f"   Function Coverage: {self._colorize_percentage(f'{function_pct:.2f}%')}")

                    # Prepare stats to return if requested
                    if return_stats:
                        if package_filter and args.packages:
                            # Return filtered stats
                            return {
                                'total_lines': total_lines if 'total_lines' in locals() else totals['lines']['count'],
                                'covered_lines': covered_lines if 'covered_lines' in locals() else totals['lines']['covered'],
                                'total_functions': total_functions if 'total_functions' in locals() else totals['functions']['count'],
                                'covered_functions': covered_functions if 'covered_functions' in locals() else totals['functions']['covered'],
                                'coverage_data': coverage_data
                            }
                        else:
                            # Return overall stats
                            return {
                                'total_lines': totals['lines']['count'],
                                'covered_lines': totals['lines']['covered'],
                                'total_functions': totals['functions']['count'],
                                'covered_functions': totals['functions']['covered'],
                                'coverage_data': coverage_data
                            }
                except Exception as e:
                    print(f"\nWARNING: Could not generate coverage summary: {e}")

            finally:
                # Clean up temp files
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                if os.path.exists(temp_json_path):
                    os.remove(temp_json_path)

        print("\nAnalysis complete")

        # Stats were collected and returned above if return_stats=True
        return None

    def _run_normal_with_stats(self, args):
        """Run coverage and return statistics for aggregation."""
        # Just call _run_normal with return_stats=True
        return self._run_normal(args, return_stats=True)

    def _print_aggregate_summary(self, stats):
        """Print aggregate coverage summary for all packages."""
        if stats['total_lines'] == 0:
            return

        line_pct = (stats['covered_lines'] / stats['total_lines'] * 100)
        function_pct = (stats['covered_functions'] / stats['total_functions'] * 100) if stats['total_functions'] > 0 else 0

        print("\n" + "="*60)
        print("AGGREGATE COVERAGE SUMMARY")
        print("="*60)

        # Show combined table if we have data
        if stats['all_coverage_data'] and HAS_TABULATE:
            # Sort by filename for consistent output
            sorted_data = sorted(stats['all_coverage_data'], key=lambda x: x['filename'])
            print("")
            print(self._format_coverage_table(sorted_data))

        print("\nOverall Coverage:")
        print(f"   Line Coverage:     {self._colorize_percentage(f'{line_pct:.2f}%')} ({stats['covered_lines']}/{stats['total_lines']} lines)")
        print(f"   Function Coverage: {self._colorize_percentage(f'{function_pct:.2f}%')} ({stats['covered_functions']}/{stats['total_functions']} functions)")

        # Show per-package summary table
        if stats.get('package_stats') and HAS_TABULATE:
            print("\n\nPer-Package Summary:")
            headers = ['Package', 'Lines', 'Line Coverage', 'Functions', 'Function Coverage']
            rows = []

            for pkg_stat in stats['package_stats']:
                row = [
                    pkg_stat['name'],
                    f"{pkg_stat['covered_lines']}/{pkg_stat['total_lines']}",
                    self._colorize_percentage(f"{pkg_stat['line_pct']:.2f}%"),
                    f"{pkg_stat['covered_functions']}/{pkg_stat['total_functions']}",
                    self._colorize_percentage(f"{pkg_stat['function_pct']:.2f}%")
                ]
                rows.append(row)

            table = tabulate(rows, headers=headers, tablefmt='simple',
                           colalign=('left', 'center', 'center', 'center', 'center'))
            print(table)

        print("")

    def _run_multiple_packages_with_aggregate_stats(self, packages, args):
        """Run coverage for multiple packages and return aggregate stats."""
        # Track aggregate statistics
        aggregate_stats = {
            'total_lines': 0,
            'covered_lines': 0,
            'total_functions': 0,
            'covered_functions': 0,
            'all_coverage_data': [],
            'package_stats': []  # Track per-package statistics
        }

        for pkg in packages:
            print(f"\n● Package: {pkg}")

            # Clean previous coverage data
            subprocess.run(["cargo", "llvm-cov", "clean", "-p", pkg],
                         capture_output=True)

            # Build package-specific args
            pkg_args = argparse.Namespace(**vars(args))
            pkg_args.packages = [pkg]
            pkg_args.workspace = False
            pkg_args.all_packages = False

            # Run coverage for this package and collect stats
            pkg_stats = self._run_normal(pkg_args, return_stats=True)
            if pkg_stats:
                aggregate_stats['total_lines'] += pkg_stats['total_lines']
                aggregate_stats['covered_lines'] += pkg_stats['covered_lines']
                aggregate_stats['total_functions'] += pkg_stats['total_functions']
                aggregate_stats['covered_functions'] += pkg_stats['covered_functions']
                
                # Store per-package summary
                pkg_line_pct = (pkg_stats['covered_lines'] / pkg_stats['total_lines'] * 100) if pkg_stats['total_lines'] > 0 else 0
                pkg_func_pct = (pkg_stats['covered_functions'] / pkg_stats['total_functions'] * 100) if pkg_stats['total_functions'] > 0 else 0
                aggregate_stats['package_stats'].append({
                    'name': pkg,
                    'line_pct': pkg_line_pct,
                    'function_pct': pkg_func_pct,
                    'covered_lines': pkg_stats['covered_lines'],
                    'total_lines': pkg_stats['total_lines'],
                    'covered_functions': pkg_stats['covered_functions'],
                    'total_functions': pkg_stats['total_functions']
                })
                
                # Prefix filenames with package name for aggregate report
                for item in pkg_stats['coverage_data']:
                    item['filename'] = f"{pkg}::{item['filename']}"
                aggregate_stats['all_coverage_data'].extend(pkg_stats['coverage_data'])
            print()

        # Print aggregate summary if we processed multiple packages
        if len(packages) > 1:
            self._print_aggregate_summary(aggregate_stats)
            
        return aggregate_stats

    def _run_multiple_packages_with_aggregate(self, packages, args):
        """Run coverage for multiple packages and show aggregate stats."""
        # Just call the stats version - it already handles printing
        return self._run_multiple_packages_with_aggregate_stats(packages, args)

    def _save_baseline(self, filename: str, stats: Dict):
        """Save coverage statistics to a baseline file."""
        # Get current git commit if available
        try:
            commit = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True).stdout.strip()
            branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                  capture_output=True, text=True).stdout.strip()
        except:
            commit = "unknown"
            branch = "unknown"
        
        baseline_data = {
            'timestamp': datetime.now().isoformat(),
            'commit': commit,
            'branch': branch,
            'packages': {},
            'overall': {
                'line_coverage': 0.0,
                'function_coverage': 0.0,
                'total_lines': stats.get('total_lines', 0),
                'covered_lines': stats.get('covered_lines', 0),
                'total_functions': stats.get('total_functions', 0),
                'covered_functions': stats.get('covered_functions', 0)
            },
            'files': {}  # Add file-level coverage details
        }
        
        # Save file-level coverage data
        if 'all_coverage_data' in stats:
            for file_data in stats['all_coverage_data']:
                baseline_data['files'][file_data['filename']] = {
                    'lines': {
                        'total': file_data['lines']['total'],
                        'covered': file_data['lines']['total'] - file_data['lines']['missed'],
                        'percent': float(file_data['lines']['percent'].rstrip('%'))
                    },
                    'functions': {
                        'total': file_data['functions']['total'],
                        'covered': file_data['functions']['total'] - file_data['functions']['missed'],
                        'percent': float(file_data['functions']['percent'].rstrip('%'))
                    }
                }
        
        # If we have package stats, use them
        if 'package_stats' in stats:
            for pkg_stat in stats['package_stats']:
                baseline_data['packages'][pkg_stat['name']] = {
                    'line_coverage': pkg_stat['line_pct'],
                    'function_coverage': pkg_stat['function_pct'],
                    'total_lines': pkg_stat['total_lines'],
                    'covered_lines': pkg_stat['covered_lines'],
                    'total_functions': pkg_stat['total_functions'],
                    'covered_functions': pkg_stat['covered_functions']
                }
        
        # Calculate overall stats
        if stats.get('total_lines', 0) > 0:
            baseline_data['overall']['line_coverage'] = (
                stats['covered_lines'] / stats['total_lines'] * 100
            )
        if stats.get('total_functions', 0) > 0:
            baseline_data['overall']['function_coverage'] = (
                stats['covered_functions'] / stats['total_functions'] * 100
            )
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        print(f"\nBaseline saved to: {filename}")
        print(f"  Branch: {branch}")
        print(f"  Commit: {commit[:8]}")
        print(f"  Overall line coverage: {baseline_data['overall']['line_coverage']:.2f}%")
        print(f"  Overall function coverage: {baseline_data['overall']['function_coverage']:.2f}%")
        print(f"  Files tracked: {len(baseline_data['files'])}")
    
    def _load_baseline(self, filename: str) -> Dict:
        """Load baseline data from file."""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Baseline file not found: {filename}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"ERROR: Invalid baseline file format: {filename}")
            sys.exit(1)
    
    def _compare_with_baseline(self, baseline: Dict, current_stats: Dict) -> bool:
        """Compare current coverage with baseline. Returns True if no regression."""
        print("\n" + "="*60)
        print("COVERAGE COMPARISON")
        print("="*60)
        
        # Show baseline info
        print(f"\nBaseline from: {baseline['branch']} ({baseline['commit'][:8]})")
        print(f"Created: {baseline['timestamp']}")
        
        # Calculate current overall coverage
        current_line_coverage = 0.0
        current_func_coverage = 0.0
        if current_stats.get('total_lines', 0) > 0:
            current_line_coverage = current_stats['covered_lines'] / current_stats['total_lines'] * 100
        if current_stats.get('total_functions', 0) > 0:
            current_func_coverage = current_stats['covered_functions'] / current_stats['total_functions'] * 100
        
        # Compare overall
        baseline_line = baseline['overall']['line_coverage']
        baseline_func = baseline['overall']['function_coverage']
        overall_line_delta = current_line_coverage - baseline_line
        overall_func_delta = current_func_coverage - baseline_func
        
        print(f"\nOverall Coverage Comparison:")
        print(f"  Line Coverage:     {baseline_line:.2f}% → {current_line_coverage:.2f}% ({self._format_delta(overall_line_delta)})")
        print(f"  Function Coverage: {baseline_func:.2f}% → {current_func_coverage:.2f}% ({self._format_delta(overall_func_delta)})")
        
        # Track regressions at different levels
        package_regressions = []
        file_regressions = []
        
        # Package-level comparison if available
        if 'package_stats' in current_stats and current_stats['package_stats']:
            print("\nPer-Package Comparison:")
            headers = ['Package', 'Lines (Base)', 'Lines (Current)', 'Delta', 'Functions (Base)', 'Functions (Current)', 'Delta']
            rows = []
            
            for pkg_stat in current_stats['package_stats']:
                pkg_name = pkg_stat['name']
                baseline_pkg = baseline['packages'].get(pkg_name, {})
                
                # Get baseline values (0 if new package)
                base_line = baseline_pkg.get('line_coverage', 0.0)
                base_func = baseline_pkg.get('function_coverage', 0.0)
                
                # Calculate deltas
                line_delta = pkg_stat['line_pct'] - base_line
                func_delta = pkg_stat['function_pct'] - base_func
                
                # Track package-level regressions
                if line_delta < 0 or func_delta < 0:
                    package_regressions.append(pkg_name)
                
                row = [
                    pkg_name,
                    f"{base_line:.2f}%",
                    f"{pkg_stat['line_pct']:.2f}%",
                    self._format_delta(line_delta),
                    f"{base_func:.2f}%",
                    f"{pkg_stat['function_pct']:.2f}%",
                    self._format_delta(func_delta)
                ]
                rows.append(row)
            
            if HAS_TABULATE:
                print(tabulate(rows, headers=headers, tablefmt='simple'))
        
        # File-level comparison if available
        if 'files' in baseline and 'all_coverage_data' in current_stats:
            print("\nFile-Level Changes:")
            file_changes = []
            
            # Create a map of current files
            current_files = {}
            for file_data in current_stats['all_coverage_data']:
                current_files[file_data['filename']] = {
                    'lines': {
                        'total': file_data['lines']['total'],
                        'covered': file_data['lines']['total'] - file_data['lines']['missed'],
                        'percent': float(file_data['lines']['percent'].rstrip('%'))
                    },
                    'functions': {
                        'total': file_data['functions']['total'],
                        'covered': file_data['functions']['total'] - file_data['functions']['missed'],
                        'percent': float(file_data['functions']['percent'].rstrip('%'))
                    }
                }
            
            # Compare files that exist in both baseline and current
            for filename, baseline_file in baseline['files'].items():
                if filename in current_files:
                    current_file = current_files[filename]
                    line_delta = current_file['lines']['percent'] - baseline_file['lines']['percent']
                    func_delta = current_file['functions']['percent'] - baseline_file['functions']['percent']
                    
                    # Only show files with significant changes
                    if abs(line_delta) > 0.1 or abs(func_delta) > 0.1:
                        file_changes.append({
                            'filename': filename,
                            'line_delta': line_delta,
                            'func_delta': func_delta,
                            'base_line': baseline_file['lines']['percent'],
                            'curr_line': current_file['lines']['percent'],
                            'base_func': baseline_file['functions']['percent'],
                            'curr_func': current_file['functions']['percent']
                        })
                        
                        # Track file-level regressions
                        if line_delta < 0 or func_delta < 0:
                            file_regressions.append(filename)
            
            # Sort by largest regression first
            file_changes.sort(key=lambda x: min(x['line_delta'], x['func_delta']))
            
            if file_changes:
                headers = ['File', 'Lines (Base)', 'Lines (Current)', 'Delta', 'Functions (Base)', 'Functions (Current)', 'Delta']
                rows = []
                
                # Show top 10 files with changes
                for change in file_changes[:10]:
                    row = [
                        change['filename'][:60] + '...' if len(change['filename']) > 60 else change['filename'],
                        f"{change['base_line']:.1f}%",
                        f"{change['curr_line']:.1f}%",
                        self._format_delta(change['line_delta']),
                        f"{change['base_func']:.1f}%",
                        f"{change['curr_func']:.1f}%",
                        self._format_delta(change['func_delta'])
                    ]
                    rows.append(row)
                
                if HAS_TABULATE:
                    print(tabulate(rows, headers=headers, tablefmt='simple'))
                
                if len(file_changes) > 10:
                    print(f"\n  ... and {len(file_changes) - 10} more files with changes")
        
        # Summary of regressions
        if package_regressions or file_regressions:
            print("\nRegression Summary:")
            if package_regressions:
                print(f"  Packages with regression: {', '.join(package_regressions)}")
            if file_regressions:
                print(f"  Files with regression: {len(file_regressions)}")
        
        # Check for regression at any level - overall, package, or file
        has_regression = overall_line_delta < 0 or overall_func_delta < 0 or bool(package_regressions) or bool(file_regressions)
        
        print("\n" + "="*60)
        if has_regression:
            print(f"{self.colors['red']}COVERAGE REGRESSION DETECTED{self.colors['reset']}")
        else:
            print(f"{self.colors['green']}COVERAGE CHECK PASSED{self.colors['reset']}")
        print("="*60)
        
        # Store regression state for programmatic access
        self._regression_detected = has_regression
        
        return not has_regression
    
    def _format_delta(self, delta: float) -> str:
        """Format coverage delta with color and sign."""
        if delta > 0:
            return f"{self.colors['green']}+{delta:.2f}%{self.colors['reset']}"
        elif delta < 0:
            return f"{self.colors['red']}{delta:.2f}%{self.colors['reset']}"
        else:
            return f"{delta:.2f}%"


def run_coverage(packages=None, all_packages=False, save_baseline=None, compare_baseline=None, 
                 fail_on_regression=False, format='summary', debug=False, **kwargs):
    """Run coverage programmatically with specified options.
    
    Returns:
        int: Exit code (0 for success, 1 for failure/regression)
    """
    # Build args namespace
    args = argparse.Namespace(
        packages=packages or [],
        all_packages=all_packages,
        save_baseline=save_baseline,
        compare_baseline=compare_baseline,
        fail_on_regression=fail_on_regression,
        format=format,
        debug=debug,
        exclude=kwargs.get('exclude', []),
        lib=kwargs.get('lib', False),
        test=kwargs.get('test', []),
        open=kwargs.get('open', False),
        no_cfg_coverage_nightly=kwargs.get('no_cfg_coverage_nightly', False),
        workspace=False
    )
    
    # Check if no package selection made
    if not args.packages and not args.all_packages:
        raise ValueError("Must specify packages or use all_packages=True")
    
    # Set workspace flag
    args.workspace = not args.packages and not args.all_packages
    
    runner = CoverageRunner()
    runner.run(args)
    
    # Return exit code based on regression check
    if args.fail_on_regression and runner._regression_detected:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Rust Code Coverage Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Package selection
    parser.add_argument('-p', '--package', dest='packages', action='append',
                       default=[], help='Run coverage for specific package(s)')
    parser.add_argument('--exclude', dest='exclude', action='append',
                       default=[], help='Exclude specific package(s)')
    parser.add_argument('--all-packages', action='store_true',
                       help='Run coverage for all packages separately')
    parser.add_argument('--lib', action='store_true',
                       help='Run only library tests')
    parser.add_argument('--test', dest='test', action='append',
                       default=[], help='Run specific test binary')

    # Output format
    parser.add_argument('--format', choices=['summary', 'html', 'json', 'lcov'],
                       default='summary', help='Output format (default: summary)')
    parser.add_argument('--open', action='store_true',
                       help='Open HTML report in browser (use with --format html)')

    # Build options
    parser.add_argument('--debug', action='store_true',
                       help='Build and test in debug mode (default: release)')
    parser.add_argument('--no-cfg-coverage-nightly', action='store_true',
                       help='Disable cfg(coverage_nightly) flag')
    
    # Baseline comparison options
    parser.add_argument('--save-baseline', metavar='FILE',
                       help='Save coverage results to baseline file')
    parser.add_argument('--compare-baseline', metavar='FILE',
                       help='Compare coverage against baseline file')
    parser.add_argument('--fail-on-regression', action='store_true',
                       help='Exit with error code if coverage decreases')

    args = parser.parse_args()

    # Check if no package selection made
    if not args.packages and not args.all_packages:
        parser.print_help()
        sys.exit(1)

    # Set workspace flag
    args.workspace = not args.packages and not args.all_packages

    runner = CoverageRunner()
    runner.run(args)


if __name__ == "__main__":
    main()
