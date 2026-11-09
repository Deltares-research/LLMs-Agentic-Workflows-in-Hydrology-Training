# Tiny long-lived Wflow runner: reads one TOML path per line on stdin, runs it, prints a marker.
# Keeping Julia alive avoids paying ~60 s of start-up/compilation for every agent run.
using Wflow
println("@@WFLOW_READY@@ ", pkgversion(Wflow)); flush(stdout)
for line in eachline(stdin)
    p = strip(line)
    isempty(p) && continue
    p == "EXIT" && break
    try
        Wflow.run(String(p))
        println("@@WFLOW_DONE@@")
    catch e
        msg = replace(sprint(showerror, e), '\n' => ' ')
        println("@@WFLOW_ERROR@@ ", first(msg, 2000))
    end
    flush(stdout)
end
