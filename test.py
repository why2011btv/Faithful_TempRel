import stopit
 
with stopit.ThreadingTimeout(5) as context_manager:
     
    # sample code we want to run...
    for i in range(10**8):
        i = i * 2
     
# Did code finish running in under 5 seconds?
if context_manager.state == context_manager.EXECUTED:
    print("COMPLETE...")
 
# Did code timeout?
elif context_manager.state == context_manager.TIMED_OUT:
    print("DID NOT FINISH...")


print("start next round")
