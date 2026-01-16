# src/recognizer.py

import os
import face_recognition # type: ignore
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np # type: ignore
import time

# import utility functions from the same folder
from src.utils import save_found_image

def _worker_process_image(filename, known_encoding, 
                          folder_path, output_folder):
    """
    Process a single image and check for a face match.
    Returns (filename, processing_time) if a match is found.
    """
    worker_start = time.time()
    
    try:
        image_path = os.path.join(folder_path, filename)
        
        # load image - converts to rgb numpy array
        unknown_image = face_recognition.load_image_file(image_path)

        # get both face locations and encodings
        face_locations = face_recognition.face_locations(unknown_image)
        unknown_encodings = face_recognition.face_encodings(unknown_image, face_locations)

        # compare each detected face with known face
        for face_encoding, face_location in zip(unknown_encodings, face_locations):
            # compare encodings using euclidean distance
            matches = face_recognition.compare_faces(
                [known_encoding], face_encoding, tolerance=0.6
            )

            if matches[0]:  # match found
                # pass the matched face location to only draw box on that face
                save_found_image(unknown_image, filename, output_folder, face_location)
                worker_time = time.time() - worker_start
                print(f"  match: {filename} (processed in {worker_time:.3f}s)")
                return (filename, worker_time)

        # no match found
        worker_time = time.time() - worker_start
        return None

    except IndexError:
        # no face detected in image
        return None
    except Exception as e:
        print(f"  error processing {filename}: {str(e)}")
        return None

class FaceRecognizer:
    """
    Handles parallel face recognition using multiprocessing with worker allocation.
    """
    def __init__(self, known_path, image_folder, output_folder):
        """
        Initialize paths and performance tracking.
        """
        self.known_path = known_path
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.known_encoding = None
        self.performance_stats = {
            'load_time': 0,
            'scan_time': 0,
            'processing_time': 0,
            'total_time': 0,
            'worker_allocation_strategy': ''
        }

    def _load_known_face(self):
        """
        Loads the known face image and extracts its encoding.
        
        Raises:
            FileNotFoundError: If known face image doesn't exist
            ValueError: If no face detected in known image
        """
        print("\nloading known face...")
        load_start = time.time()
        
        if not os.path.exists(self.known_path):
            raise FileNotFoundError(f"known face image not found: {self.known_path}")
        
        known_image = face_recognition.load_image_file(self.known_path)
        encodings = face_recognition.face_encodings(known_image)
        
        if not encodings:
            raise ValueError(f"no face detected in: {self.known_path}")
        
        self.known_encoding = encodings[0]
        self.performance_stats['load_time'] = time.time() - load_start
        print(f"  known face loaded ({self.performance_stats['load_time']:.3f}s)")

    def _get_image_files(self):
        """
        Return a list of valid image files from the image folder.
        """
        print("\nscanning image folder...")
        scan_start = time.time()
        
        if not os.path.exists(self.image_folder):
            raise FileNotFoundError(f"image folder not found: {self.image_folder}")
        
        # get all files (filters out directories)
        filenames = [f.name for f in os.scandir(self.image_folder) if f.is_file()]
        
        # filter for image extensions only
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        filenames = [f for f in filenames 
                    if os.path.splitext(f.lower())[1] in valid_extensions]
        
        self.performance_stats['scan_time'] = time.time() - scan_start
        print(f"  found {len(filenames)} images ({self.performance_stats['scan_time']:.3f}s)")
        
        return filenames

    def _determine_optimal_processes(self, num_images):
        """
        Determine optimal worker count and chunk size based on dataset size.
        
        Returns:
            Tuple[int, int]: (num_processes, chunksize)
        """
        available_cores = cpu_count()
        
        # very small dataset (1-10 images)
        if num_images <= 10:
            # overhead of creating processes outweighs benefits
            optimal = min(max(1, num_images // 3), 4)
            chunksize = 1
            strategy = "very small dataset strategy (overhead minimization)"
            print(f"strategy: very small dataset")
            print(f"   using {optimal} workers (1 per ~3 images to minimize overhead)")
            print(f"   chunk size: {chunksize}")
        
        # small dataset (11-50 images)
        elif num_images <= 50:
            optimal = min(num_images // 4, available_cores // 2)
            optimal = max(optimal, 2)
            chunksize = max(1, num_images // (optimal * 3))
            strategy = "small dataset strategy (proportional scaling)"
            print(f"strategy: small dataset")
            print(f"   using {optimal} workers (~1 per 4 images)")
            print(f"   chunk size: {chunksize}")
        
        # medium dataset (51-200 images)
        elif num_images <= 200:
            optimal = min(num_images // 3, int(available_cores * 0.75))
            optimal = max(optimal, 4)
            chunksize = 2
            strategy = "medium dataset strategy (balanced utilization)"
            print(f"strategy: medium dataset")
            print(f"   using {optimal} workers (~75% of cores)")
            print(f"   chunk size: {chunksize}")
        
        # large dataset (201-1000 images)
        elif num_images <= 1000:
            optimal = min(num_images // 5, available_cores)
            chunksize = max(2, num_images // (optimal * 4))
            strategy = "large dataset strategy (high parallelism)"
            print(f"strategy: large dataset")
            print(f"   using {optimal} workers (maximizing core usage)")
            print(f"   chunk size: {chunksize}")
        
        # very large dataset (1000+ images)
        else:
            optimal = available_cores
            chunksize = max(3, num_images // (optimal * 10))
            strategy = "very large dataset strategy (full parallelism)"
            print(f"strategy: very large dataset")
            print(f"   using all {optimal} cores")
            print(f"   chunk size: {chunksize} (optimized for throughput)")
        
        # don't create more workers than images
        optimal = min(optimal, num_images)
        
        # store strategy for reporting
        self.performance_stats['worker_allocation_strategy'] = strategy
        
        return optimal, chunksize

    def _calculate_workload_distribution(self, num_images, num_processes, 
                                         chunksize):
        """
        Calculate and display how work will be distributed across workers.
        """
        print(f"\nworkload distribution analysis:")
        print(f"   total images:     {num_images}")
        print(f"   worker processes: {num_processes}")
        print(f"   chunk size:       {chunksize}")
        
        # calculate theoretical distribution
        total_chunks = (num_images + chunksize - 1) // chunksize
        chunks_per_worker_avg = total_chunks / num_processes
        
        print(f"   total chunks:     {total_chunks}")
        print(f"   avg chunks/worker: {chunks_per_worker_avg:.2f}")
        
        # estimate based on dynamic scheduling
        images_per_worker_base = num_images // num_processes
        remainder = num_images % num_processes
        
        if remainder > 0:
            print(f"\ndynamic distribution (estimated):")
            print(f"   {num_processes - remainder} workers: ~{images_per_worker_base} images")
            print(f"   {remainder} workers: ~{images_per_worker_base + 1} images")
        else:
            print(f"\nbalanced distribution:")
            print(f"   each worker: ~{images_per_worker_base} images")
        
        # calculate efficiency metrics
        overhead_ratio = (num_processes * 0.05) / num_images
        print(f"\nestimated process overhead: {overhead_ratio*100:.2f}%")
        
        if overhead_ratio > 0.1:
            print(f"   warning: high overhead ratio - consider fewer workers")
        else:
            print(f"   good overhead/work ratio")

    def run_parallel_recognition(self):
        """
        Run face recognition on all images using parallel processing.
        """
        total_start = time.time()
        
        # load known face
        self._load_known_face()
        
        # get all image files
        filenames = self._get_image_files()
        total_images = len(filenames)
        
        if total_images == 0:
            print("no images found to process")
            return [], 0
        
        # determine optimal configuration
        print("\nconfiguring worker allocation...")
        print(f"   available cpu cores: {cpu_count()}")
        print(f"   dataset size: {total_images} images")
        print()
        
        num_processes, chunksize = self._determine_optimal_processes(total_images)
        
        # show workload distribution
        self._calculate_workload_distribution(total_images, num_processes, chunksize)
        
        # create worker function with fixed parameters
        process_func = partial(
            _worker_process_image,
            known_encoding=self.known_encoding,
            folder_path=self.image_folder,
            output_folder=self.output_folder
        )

        # execute parallel processing
        print(f"\nprocessing {total_images} images with {num_processes} workers...")
        print(f"   (dynamic load balancing with chunk size: {chunksize})")
        
        processing_start = time.time()
        
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_func, filenames, chunksize=chunksize)
        
        self.performance_stats['processing_time'] = time.time() - processing_start
        
        # filter and collect results
        print("\ncollecting results...")
        
        matched_results = [r for r in results if r is not None]
        matched_files = [filename for filename, _ in matched_results]
        
        if matched_results:
            processing_times = [proc_time for _, proc_time in matched_results]
            avg_match_time = sum(processing_times) / len(processing_times)
            print(f"   average time per matched image: {avg_match_time:.3f}s")
        
        self.performance_stats['total_time'] = time.time() - total_start
        self.performance_stats['num_workers'] = num_processes
        self.performance_stats['chunksize'] = chunksize
        
        return matched_files, total_images
    
    def get_performance_report(self):
        """
        Return a formatted performance summary.
        """
        report = "\n" + "-"*70 + "\n"
        report += "performance breakdown\n"
        report += "-"*70 + "\n"
        report += f"known face loading:    {self.performance_stats['load_time']:.3f}s\n"
        report += f"image folder scanning: {self.performance_stats['scan_time']:.3f}s\n"
        report += f"parallel processing:   {self.performance_stats['processing_time']:.3f}s\n"
        report += f"total execution time:  {self.performance_stats['total_time']:.3f}s\n\n"
        
        report += "worker configuration\n"
        report += "-"*70 + "\n"
        report += f"strategy: {self.performance_stats.get('worker_allocation_strategy', 'n/a')}\n"
        report += f"workers used: {self.performance_stats.get('num_workers', 'n/a')}\n"
        report += f"chunk size: {self.performance_stats.get('chunksize', 'n/a')}\n"
        
        return report