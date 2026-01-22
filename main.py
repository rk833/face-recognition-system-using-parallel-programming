"""
parallel face recognition system (task 1.4)

features intelligent dynamic worker allocation based on dataset size.
automatically optimizes worker count and chunk size for maximum efficiency.
"""

import os
import time
from src.recognizer import FaceRecognizer

def print_header(title):
    """print formatted section header"""
    print("\n" + "-"*70)
    print(title.center(70))
    print("-"*70)

def print_section(title):
    """print formatted subsection"""
    print(f"\n{title}")

def run_serial_comparison(known_image_path, image_folder_path, 
                         output_folder_path):
    """
    run serial version for performance comparison.
    
    returns:
        serial execution time in seconds
    """
    import face_recognition
    
    print_section("serial processing (for comparison)")
    
    start_time = time.time()
    
    # load known face
    print("loading known face...")
    known_image = face_recognition.load_image_file(known_image_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]
    
    # get all images
    print("scanning images...")
    filenames = [f.name for f in os.scandir(image_folder_path) if f.is_file()]
    total_images = len(filenames)
    
    print(f"processing {total_images} images serially (one at a time)...")
    matches = 0
    
    # process each image one by one (serial)
    for i, filename in enumerate(filenames, 1):
        try:
            image_path = os.path.join(image_folder_path, filename)
            unknown_image = face_recognition.load_image_file(image_path)
            unknown_encodings = face_recognition.face_encodings(unknown_image)
            
            for unknown_encoding in unknown_encodings:
                result = face_recognition.compare_faces(
                    [known_encoding], unknown_encoding, tolerance=0.6
                )
                if result[0]:
                    matches += 1
                    print(f"  [{i}/{total_images}] match: {filename}")
                    break
        except:
            pass
    
    serial_time = time.time() - start_time
    
    print(f"\nserial processing complete")
    print(f"  total time: {serial_time:.2f}s")
    print(f"  matches found: {matches}")
    
    return serial_time

def demonstrate_scalability():
    """
    demonstrate how worker allocation scales with dataset size.
    """
    print_header("Dynamic worker allocation demonstration")
    print("\nThis shows how the system adapts to different dataset sizes:\n")
    
    from multiprocessing import cpu_count
    cores = cpu_count()
    
    # simulate different dataset sizes
    scenarios = [
        (5, "very small dataset"),
        (25, "small dataset"),
        (100, "medium dataset"),
        (500, "large dataset"),
        (2000, "very large dataset"),
    ]
    
    print(f"{'dataset size':<20} {'workers':<12} {'chunk size':<15} {'strategy'}")
    print("-" * 70)
    
    for size, description in scenarios:
        # simulate the logic from _determine_optimal_processes
        if size <= 10:
            workers = min(max(1, size // 3), 4)
            chunk = 1
            strategy = "overhead min"
        elif size <= 50:
            workers = min(size // 4, cores // 2)
            workers = max(workers, 2)
            chunk = max(1, size // (workers * 3))
            strategy = "proportional"
        elif size <= 200:
            workers = min(size // 3, int(cores * 0.75))
            workers = max(workers, 4)
            chunk = 2
            strategy = "balanced"
        elif size <= 1000:
            workers = min(size // 5, cores)
            chunk = max(2, size // (workers * 4))
            strategy = "high parallel"
        else:
            workers = cores
            chunk = max(3, size // (workers * 10))
            strategy = "full parallel"
        
        workers = min(workers, size)
        
        print(f"{size:>4} images ({description:<12}) {workers:>2} workers    "
              f"chunk={chunk:<10} {strategy}")

def main():
    """
    main execution function with intelligent dynamic worker allocation.
    """
    
    print_header("Parallel face recognition system")
    print("\nfeatures:")
    print("  dynamic worker allocation based on dataset size")
    print("  automatic chunk size optimization")
    print("  overhead-aware scaling strategies")
    print("  load balancing with work stealing")
    print("  performance profiling and comparison")
    
    # show scalability demonstration
    demonstrate_scalability()
    
    # configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    known_image_path = os.path.join(base_dir, "known_man.jpg")
    image_folder_path = os.path.join(base_dir, "imageset")
    output_folder_path = os.path.join(base_dir, "faces_detected")
    
    # validate paths
    if not os.path.exists(known_image_path):
        print(f"\nerror: known face image not found at: {known_image_path}")
        return
    
    if not os.path.exists(image_folder_path):
        print(f"\nerror: image folder not found at: {image_folder_path}")
        return

    # user choice: serial or parallel comparison
    print_section("performance comparison mode")
    print("\ndo you want to run serial comparison?")
    print("  y = yes, compare with serial version (recommended for datasets < 100)")
    print("  n = no, run optimized parallel only")
    
    choice = input("\nyour choice [y/n]: ").strip().lower()
    
    serial_time = None
    if choice == 'y':
        try:
            serial_time = run_serial_comparison(
                known_image_path, image_folder_path, output_folder_path
            )
        except Exception as e:
            print(f"serial comparison failed: {e}")
            print("  continuing with parallel processing...")
    
    # parallel processing with dynamic workers
    print_header("Parallel processing")
    
    try:
        # initialize recognizer
        recognizer = FaceRecognizer(
            known_image_path, 
            image_folder_path, 
            output_folder_path
        )
        
        # run parallel recognition with dynamic worker allocation
        parallel_start = time.time()
        matched_files, total_images = recognizer.run_parallel_recognition()
        parallel_time = time.time() - parallel_start
        
        # results summary
        print_header("processing complete")
        
        print(f"\nprocessing statistics:")
        print(f"  total images processed:  {total_images}")
        print(f"  matches found:           {len(matched_files)}")
        if total_images > 0:
            print(f"  match rate:              {len(matched_files)/total_images*100:.1f}%")
        
        if total_images > 0:
            print(f"\ntiming analysis:")
            print(f"  parallel time:           {parallel_time:.3f}s")
            print(f"  avg per image:           {parallel_time/total_images:.3f}s")
            print(f"  throughput:              {total_images/parallel_time:.2f} images/sec")
            
        # performance comparison
        if serial_time:
            speedup = serial_time / parallel_time
            efficiency = (speedup / os.cpu_count()) * 100
            
            print(f"\nspeedup analysis:")
            print(f"  serial time:             {serial_time:.3f}s")
            print(f"  parallel time:           {parallel_time:.3f}s")
            print(f"  speedup:                 {speedup:.2f}x faster")
            print(f"  parallel efficiency:     {efficiency:.1f}%")
            print(f"  time saved:              {serial_time - parallel_time:.3f}s")
            
            # efficiency rating
            if efficiency >= 80:
                rating = "excellent"
            elif efficiency >= 60:
                rating = "good"
            elif efficiency >= 40:
                rating = "fair"
            else:
                rating = "poor - consider optimization"
            print(f"  efficiency rating:       {rating}")
        
        # performance breakdown
        print(recognizer.get_performance_report())
        
        # matched files list
        if matched_files:
            print(f"\nmatched files ({len(matched_files)}):")
            for i, filename in enumerate(matched_files, 1):
                print(f"  {i:2d}. {filename}")
        else:
            print("\nno matches found in the dataset.")
        
        # system info
        print(f"\nsystem information:")
        print(f"  cpu cores available:     {os.cpu_count()}")
        print(f"  cpu cores used:          {recognizer.performance_stats.get('num_workers', 'n/a')}")
        print(f"  utilization:             {(recognizer.performance_stats.get('num_workers', 0) / os.cpu_count() * 100):.1f}%")
        print(f"  parallelization model:   data decomposition (mimd)")
        print(f"  memory model:            distributed memory")
        print(f"  load balancing:          dynamic (work stealing)")
        print(f"  chunk size:              {recognizer.performance_stats.get('chunksize', 'n/a')}")
        
        print("All processing complete!".center(70))
        
    except Exception as e:
        print(f"\nfatal error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()