#!/usr/bin/env python3
"""
Receipt Processing Pipeline
Integrates main2.py and ai.py in a seamless workflow
"""

import subprocess
import sys
import os
import logging
import argparse
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReceiptPipeline:
    """Pipeline that combines main2.py and ai.py for complete receipt processing."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.check_dependencies()
    
    def check_dependencies(self):
        """Check if required scripts exist."""
        required_files = ['main2.py', 'ai.py']
        missing_files = []
        
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            raise FileNotFoundError(f"Required files not found: {missing_files}")
        
        logger.info("All required scripts found ✓")
    
    def run_image_processing(self, input_image, output_pdf, debug=False):
        """Run main2.py to process image and create PDF."""
        try:
            logger.info("=== Step 1: Image Processing (main2.py) ===")
            logger.info(f"Processing: {input_image} → {output_pdf}")
            
            # Build command for main2.py
            cmd = ['python3', 'main2.py', '--input', input_image, '--output-pdf', output_pdf]
            
            if debug:
                cmd.append('--debug')
            
            # Run main2.py with inherited environment
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=os.environ.copy())
            
            if result.returncode == 0:
                logger.info("✅ Image processing completed successfully")
                logger.info(f"PDF created: {output_pdf}")
                return True
            else:
                logger.error("❌ Image processing failed")
                logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Image processing timed out")
            return False
        except Exception as e:
            logger.error(f"Error running image processing: {str(e)}")
            return False
    
    def run_ai_parsing(self, input_pdf, output_json):
        """Run ai.py to parse PDF and create JSON."""
        try:
            logger.info("=== Step 2: AI Parsing (ai.py) ===")
            logger.info(f"Parsing: {input_pdf} → {output_json}")
            
            # Build command for ai.py
            cmd = ['python3', 'ai.py', input_pdf]
            
            # Prepare environment with explicit GOOGLE_APPLICATION_CREDENTIALS
            env = os.environ.copy()
            if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
                env['GOOGLE_APPLICATION_CREDENTIALS'] = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
                logger.info(f"Using credentials: {env['GOOGLE_APPLICATION_CREDENTIALS']}")
            
            # Run ai.py with inherited environment (including GOOGLE_APPLICATION_CREDENTIALS)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            
            if result.returncode == 0:
                logger.info("✅ AI parsing completed successfully")
                
                # Check if parsed_receipt.json was created (default output from ai.py)
                if Path('parsed_receipt.json').exists():
                    # Rename to desired output name
                    if output_json != 'parsed_receipt.json':
                        os.rename('parsed_receipt.json', output_json)
                    logger.info(f"JSON created: {output_json}")
                    return True
                else:
                    logger.error("AI parsing completed but no JSON output found")
                    return False
            else:
                logger.error("❌ AI parsing failed")
                logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("AI parsing timed out")
            return False
        except Exception as e:
            logger.error(f"Error running AI parsing: {str(e)}")
            return False
    
    def process_receipt(self, input_image, output_json='pipeline_receipt.json', 
                       temp_pdf='temp_pipeline.pdf', debug=False, keep_pdf=False):
        """Run the complete pipeline: image → PDF → JSON."""
        try:
            logger.info("🚀 Starting Receipt Processing Pipeline")
            logger.info(f"Input: {input_image}")
            logger.info(f"Output: {output_json}")
            
            # Step 1: Image processing
            if not self.run_image_processing(input_image, temp_pdf, debug):
                logger.error("Pipeline failed at image processing step")
                return None
            
            # Verify PDF was created
            if not Path(temp_pdf).exists():
                logger.error(f"PDF not found: {temp_pdf}")
                return None
            
            pdf_size = os.path.getsize(temp_pdf) / 1024
            logger.info(f"PDF size: {pdf_size:.1f} KB")
            
            # Step 2: AI parsing
            if not self.run_ai_parsing(temp_pdf, 'temp_parsed.json'):
                logger.error("Pipeline failed at AI parsing step")
                return None
            
            # Load the new result
            try:
                with open('temp_parsed.json', 'r', encoding='utf-8') as f:
                    new_result = json.load(f)
                
                # Add metadata to the new result
                import datetime
                new_result['processed_at'] = datetime.datetime.now().isoformat()
                new_result['source_image'] = input_image
                
                # Load existing results if file exists
                all_results = []
                if Path(output_json).exists():
                    try:
                        with open(output_json, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                            # Handle both single object and array formats
                            if isinstance(existing_data, list):
                                all_results = existing_data
                            else:
                                all_results = [existing_data]
                        logger.info(f"Found {len(all_results)} existing receipt(s)")
                    except Exception as e:
                        logger.warning(f"Could not load existing results: {e}")
                        all_results = []
                
                # Add new result to the collection
                all_results.append(new_result)
                
                # Save combined results
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, separators=(',', ':'), ensure_ascii=False, indent=2)
                
                logger.info("✅ Pipeline completed successfully!")
                logger.info(f"📊 Total receipts in file: {len(all_results)}")
                
                # Clean up temporary files
                try:
                    os.remove('temp_parsed.json')
                    if not keep_pdf:
                        os.remove(temp_pdf)
                        logger.info("Cleaned up temporary files")
                    else:
                        logger.info(f"Keeping PDF: {temp_pdf}")
                except:
                    pass
                
                return new_result
                
            except Exception as e:
                logger.error(f"Error processing results: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return None
    
    def print_results(self, result):
        """Print formatted results."""
        if not result:
            print("❌ No results to display")
            return
        
        print("\n" + "="*50)
        print("📋 RECEIPT PROCESSING RESULTS")
        print("="*50)
        
        # Store info
        if result.get('store_name'):
            print(f"🏪 Store: {result['store_name']}")
        if result.get('store_address'):
            print(f"📍 Address: {result['store_address']}")
        
        # Receipt category
        if result.get('receipt_category'):
            print(f"📂 Category: {result['receipt_category']}")
        
        # Transaction info
        if result.get('date'):
            print(f"📅 Date: {result['date']}")
        if result.get('time'):
            print(f"⏰ Time: {result['time']}")
        if result.get('bill_number'):
            print(f"🧾 Bill #: {result['bill_number']}")
        
        # Processing info
        if result.get('processed_at'):
            print(f"⏱️ Processed: {result['processed_at']}")
        if result.get('source_image'):
            print(f"📸 Source: {result['source_image']}")
        
        # Payment info
        if result.get('payment_method'):
            print(f"💳 Payment: {result['payment_method']}")
        if result.get('total_amount'):
            currency = result.get('currency', '')
            print(f"💰 Total: {currency}{result['total_amount']}")
        
        # Items
        if result.get('items'):
            print(f"\n🛍️ Items ({len(result['items'])}):")
            for i, item in enumerate(result['items'], 1):
                name = item.get('item_name', 'Unknown')
                qty = item.get('quantity', '1')
                price = item.get('total_price', '0')
                category = item.get('category', 'Others')
                print(f"  {i}. {name} (x{qty}) - {price} [{category}]")
        
        # Summary
        if result.get('Summary'):
            print(f"\n📝 Summary: {result['Summary']}")
        
        print("="*50)
    
    def print_all_results(self, json_file):
        """Print all results from the JSON file."""
        if not Path(json_file).exists():
            print("❌ No results file found")
            return
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                print(f"\n📊 SHOWING ALL {len(data)} RECEIPTS")
                print("="*60)
                for i, receipt in enumerate(data, 1):
                    print(f"\n🧾 RECEIPT #{i}")
                    print("-" * 30)
                    self.print_receipt_summary(receipt)
                print("="*60)
            else:
                print("\n📊 SHOWING 1 RECEIPT")
                self.print_results(data)
                
        except Exception as e:
            print(f"❌ Error reading results: {e}")
    
    def print_receipt_summary(self, receipt):
        """Print a summary of a single receipt."""
        if receipt.get('store_name'):
            print(f"🏪 {receipt['store_name']}")
        if receipt.get('receipt_category'):
            print(f"📂 {receipt['receipt_category']}")
        if receipt.get('date') and receipt.get('time'):
            print(f"📅 {receipt['date']} at {receipt['time']}")
        elif receipt.get('date'):
            print(f"📅 {receipt['date']}")
        if receipt.get('total_amount'):
            currency = receipt.get('currency', '')
            print(f"💰 {currency}{receipt['total_amount']}")
        if receipt.get('source_image'):
            print(f"📸 {receipt['source_image']}")
        if receipt.get('processed_at'):
            print(f"⏱️ {receipt['processed_at'][:19].replace('T', ' ')}")  # Format timestamp

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Receipt Processing Pipeline')
    parser.add_argument('--input', help='Input image path')
    parser.add_argument('--output', default='pipeline_receipt.json', help='Output JSON path')
    parser.add_argument('--keep-pdf', action='store_true', help='Keep temporary PDF file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--show-all', action='store_true', help='Show all stored receipts')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ReceiptPipeline()
    
    # Show all results if requested
    if args.show_all:
        pipeline.print_all_results(args.output)
        return
    
    # Validate input for processing
    if not args.input:
        parser.error("--input is required unless using --show-all")
    
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    try:
        # Process receipt
        result = pipeline.process_receipt(
            input_image=args.input,
            output_json=args.output,
            debug=args.debug,
            keep_pdf=args.keep_pdf
        )
        
        if result:
            pipeline.print_results(result)
            print(f"\n📁 Results saved to: {args.output}")
        else:
            logger.error("Pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 