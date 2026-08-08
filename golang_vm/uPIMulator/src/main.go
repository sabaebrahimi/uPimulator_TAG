package main

import (
	"fmt"
	"os"
	"path/filepath"
	"uPIMulator/src/device/compiler"
	"uPIMulator/src/device/linker"
	"uPIMulator/src/host/interpreter"
	"uPIMulator/src/misc"
	"uPIMulator/src/program"
	"uPIMulator/src/system"
)

func main() {
	command_line_parser := InitCommandLineParser()
	command_line_parser.Parse(os.Args)

	if command_line_parser.IsArgSet("help") {
		fmt.Printf("%s", command_line_parser.StringifyHelpMsgs())
	} else {
		command_line_validator := new(misc.CommandLineValidator)
		command_line_validator.Init(command_line_parser)
		command_line_validator.Validate()

		config_loader := new(misc.ConfigLoader)
		config_loader.Init()

		config_validator := new(misc.ConfigValidator)
		config_validator.Init(config_loader)
		config_validator.Validate()

		bin_dirpath := command_line_parser.StringParameter("bin_dirpath")

		if _, err := os.Stat(bin_dirpath); !os.IsNotExist(err) {
			remove_err := os.RemoveAll(bin_dirpath)
			if remove_err != nil {
				panic(remove_err)
			}
		}

		mkdir_err := os.MkdirAll(bin_dirpath, os.ModePerm)
		if mkdir_err != nil {
			panic(mkdir_err)
		}

		args_filepath := filepath.Join(bin_dirpath, "args.txt")
		options_filepath := filepath.Join(bin_dirpath, "options.txt")

		args_file_dumper := new(misc.FileDumper)
		args_file_dumper.Init(args_filepath)
		args_file_dumper.WriteLines([]string{command_line_parser.StringifyArgs()})

		options_file_dumper := new(misc.FileDumper)
		options_file_dumper.Init(options_filepath)
		options_file_dumper.WriteLines([]string{command_line_parser.StringifyOptions()})

		// skip_compile != 0 reuses prebuilt DPU/SDK assembly under benchmark/build and
		// sdk/build instead of invoking the UPMEM SDK toolchain in Docker. Used by the
		// PIM-DL co-simulation flow, which compiles the DPU kernels with a local SDK
		// (see tools/build_dpu_local.sh) and does not have Docker available.
		if command_line_parser.IntParameter("skip_compile") == 0 {
			compiler_ := new(compiler.Compiler)
			compiler_.Init(command_line_parser)
			compiler_.Compile()
		}

		linker_ := new(linker.Linker)
		linker_.Init(command_line_parser)
		linker_.Link()

		program.RunPimdlMramPatchIfPresent(
			command_line_parser.StringParameter("bin_dirpath"),
			command_line_parser.StringParameter("pimdl_patch_dirpath"),
		)

		task := new(program.Task)
		task.Init(command_line_parser)

		interpreter_ := new(interpreter.Interpreter)
		interpreter_.Init(command_line_parser, task.SysUsedMramEnd())
		interpreter_.Interpret()

		app := new(program.App)
		app.Init(command_line_parser)

		system_ := new(system.System)
		system_.Init(command_line_parser)
		system_.Simulate(app, task)
		system_.Dump()
		system_.Fini()
	}
}

func InitCommandLineParser() *misc.CommandLineParser {
	command_line_parser := new(misc.CommandLineParser)
	command_line_parser.Init()

	// NOTE(dongjae.lee@kaist.ac.kr): Explanation of verbose levels
	// level 0: Only prints simulation output
	// level 1: level 0 + prints UPMEM instruction executed per each logic cycle
	// level 2: level 1 + prints host VM's stack and UPMEM register file values per each logic cycle
	command_line_parser.AddOption(misc.INT, "verbose", "1", "verbosity of the simulation")

	command_line_parser.AddOption(misc.STRING, "benchmark", "GEMV", "benchmark name")

	command_line_parser.AddOption(misc.INT, "num_channels", "1", "number of PIM memory channels")
	command_line_parser.AddOption(
		misc.INT,
		"num_ranks_per_channel",
		"1",
		"number of ranks per channel",
	)
	command_line_parser.AddOption(misc.INT, "num_dpus_per_rank", "4", "number of DPUs per rank")

	command_line_parser.AddOption(misc.INT, "num_vm_channels", "1", "number of VM memory channels")
	command_line_parser.AddOption(
		misc.INT,
		"num_vm_ranks_per_channel",
		"2",
		"number of VM ranks per channel",
	)
	command_line_parser.AddOption(
		misc.INT,
		"num_vm_banks_per_rank",
		"16",
		"number of VM banks per rank",
	)

	command_line_parser.AddOption(misc.INT, "num_tasklets", "16", "number of tasklets")
	command_line_parser.AddOption(misc.STRING, "data_prep_params", "65536",
		"data preparation parameter")

	command_line_parser.AddOption(
		misc.STRING,
		"root_dirpath",
		"/home/via/uPIMulator/golang_vm/uPIMulator",
		"path to the root directory",
	)

	command_line_parser.AddOption(misc.STRING, "bin_dirpath",
		"/home/via/uPIMulator/golang_vm/uPIMulator/bin", "path to the bin directory")

	// PIM-DL co-simulation: directory holding pimdl_mram_patch.json + segment .bin
	// files used to splice PIM-DL's dumped LUT/index data into the linked mram.bin.
	// Kept separate from bin_dirpath because bin_dirpath is wiped at startup.
	// Empty (default) falls back to bin_dirpath for backward compatibility.
	command_line_parser.AddOption(misc.STRING, "pimdl_patch_dirpath", "",
		"path to the PIM-DL mram patch directory (manifest + segments)")

	command_line_parser.AddOption(misc.STRING, "pimdl_xfer_trace_path", "",
		"path to the PIM-DL JSONL transfer trace (defaults to the benchmark boundary trace in pimdl_patch_dirpath)")

	command_line_parser.AddOption(misc.INT, "skip_compile", "0",
		"skip the Docker/UPMEM compile step and reuse prebuilt benchmark/build + sdk/build assembly")

	command_line_parser.AddOption(misc.INT, "logic_frequency", "350", "DPU logic frequency in MHz")
	command_line_parser.AddOption(misc.INT, "memory_frequency", "2400",
		"DPU MRAM frequency in MHz")

	command_line_parser.AddOption(misc.INT, "num_pipeline_stages", "14",
		"number of DPU logic pipeline stages")
	command_line_parser.AddOption(misc.INT, "num_revolver_scheduling_cycles", "11",
		"number of DPU logic revolver scheduling cycles")

	command_line_parser.AddOption(misc.INT, "wordline_size", "1024",
		"row buffer size per single DPU's MRAM in bytes")
	command_line_parser.AddOption(misc.INT, "min_access_granularity", "8",
		"DPU MRAM's minimum access granularity in bytes")

	command_line_parser.AddOption(misc.INT, "reorder_window_size", "256", "FR-FCFS reorder window size")

	command_line_parser.AddOption(
		misc.INT,
		"t_rcd",
		"32",
		"DPU MRAM t_rcd timing parameter [cycle]",
	)
	command_line_parser.AddOption(
		misc.INT,
		"t_ras",
		"78",
		"DPU MRAM t_ras timing parameter [cycle]",
	)
	command_line_parser.AddOption(misc.INT, "t_rp", "32", "DPU MRAM t_rp timing parameter [cycle]")
	command_line_parser.AddOption(misc.INT, "t_cl", "32", "DPU MRAM t_cl timing parameter [cycle]")
	command_line_parser.AddOption(misc.INT, "t_bl", "8", "DPU MRAM t_bl timing parameter [cycle]")

	return command_line_parser
}
