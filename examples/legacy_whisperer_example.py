"""
CRONOS AI - Legacy System Whisperer Example
Demonstrates usage of the Legacy System Whisperer for protocol analysis and modernization.
"""

import asyncio
from ai_engine.llm.legacy_whisperer import create_legacy_whisperer, AdapterLanguage


async def example_reverse_engineering():
    """Example: Reverse engineer a legacy protocol."""
    print("=" * 60)
    print("Example 1: Protocol Reverse Engineering")
    print("=" * 60)

    # Initialize whisperer
    whisperer = await create_legacy_whisperer()

    # Create sample traffic (simulated legacy protocol)
    # Format: Magic number (4 bytes) + Length (2 bytes) + Payload
    traffic_samples = []
    for i in range(25):
        magic = b"\x42\x43\x44\x45"  # "BCDE"
        length = (20 + i).to_bytes(2, "big")
        payload = bytes([i % 256] * (20 + i))
        traffic_samples.append(magic + length + payload)

    print(f"\nAnalyzing {len(traffic_samples)} traffic samples...")

    # Reverse engineer the protocol
    spec = await whisperer.reverse_engineer_protocol(
        traffic_samples=traffic_samples,
        system_context="Legacy financial transaction protocol from 1990s mainframe system",
    )

    # Display results
    print(f"\n✓ Protocol Analysis Complete!")
    print(f"  Protocol Name: {spec.protocol_name}")
    print(f"  Version: {spec.version}")
    print(f"  Complexity: {spec.complexity.value}")
    print(f"  Confidence: {spec.confidence_score:.2%}")
    print(f"  Analysis Time: {spec.analysis_time:.2f}s")
    print(f"\n  Detected Fields: {len(spec.fields)}")
    for field in spec.fields[:5]:  # Show first 5 fields
        print(f"    - {field.name}: {field.field_type} at offset {field.offset}")

    print(f"\n  Detected Patterns: {len(spec.patterns)}")
    for pattern in spec.patterns:
        print(f"    - {pattern.pattern_type}: {pattern.description}")

    print(f"\n  Message Types: {len(spec.message_types)}")

    await whisperer.shutdown()
    return spec


async def example_adapter_generation(spec):
    """Example: Generate protocol adapter code."""
    print("\n" + "=" * 60)
    print("Example 2: Adapter Code Generation")
    print("=" * 60)

    whisperer = await create_legacy_whisperer()

    # Cache the specification
    cache_key = whisperer._generate_cache_key([], "")
    whisperer._cache_specification(cache_key, spec)

    print(f"\nGenerating Python adapter for {spec.protocol_name} → REST...")

    # Generate adapter code
    adapter = await whisperer.generate_adapter_code(
        legacy_protocol=spec, target_protocol="REST", language=AdapterLanguage.PYTHON
    )

    # Display results
    print(f"\n✓ Adapter Generation Complete!")
    print(f"  Adapter ID: {adapter.adapter_id}")
    print(f"  Language: {adapter.language.value}")
    print(f"  Code Quality Score: {adapter.code_quality_score:.2%}")
    print(f"  Generation Time: {adapter.generation_time:.2f}s")
    print(
        f"  Dependencies: {', '.join(adapter.dependencies) if adapter.dependencies else 'None'}"
    )

    print(f"\n  Generated Files:")
    print(f"    - Adapter Code: {len(adapter.adapter_code)} characters")
    print(f"    - Test Code: {len(adapter.test_code)} characters")
    print(f"    - Documentation: {len(adapter.documentation)} characters")

    # Save generated files
    print(f"\n  Saving generated files...")
    with open("/tmp/legacy_adapter.py", "w") as f:
        f.write(adapter.adapter_code)
    print(f"    ✓ Saved: /tmp/legacy_adapter.py")

    with open("/tmp/test_legacy_adapter.py", "w") as f:
        f.write(adapter.test_code)
    print(f"    ✓ Saved: /tmp/test_legacy_adapter.py")

    with open("/tmp/adapter_documentation.md", "w") as f:
        f.write(adapter.documentation)
    print(f"    ✓ Saved: /tmp/adapter_documentation.md")

    with open("/tmp/adapter_config.ini", "w") as f:
        f.write(adapter.configuration_template)
    print(f"    ✓ Saved: /tmp/adapter_config.ini")

    await whisperer.shutdown()


async def example_behavior_explanation():
    """Example: Explain legacy system behavior."""
    print("\n" + "=" * 60)
    print("Example 3: Legacy Behavior Explanation")
    print("=" * 60)

    whisperer = await create_legacy_whisperer()

    behavior = """
    The system uses fixed-width records with EBCDIC encoding for all data storage.
    Batch processing runs overnight with no real-time capabilities.
    All transactions are logged to sequential files on tape storage.
    """

    context = {
        "system_type": "mainframe",
        "era": "1980s",
        "industry": "banking",
        "criticality": "high",
        "users": 5000,
    }

    print(f"\nAnalyzing legacy behavior...")
    print(f"Behavior: {behavior.strip()}")

    # Explain the behavior
    explanation = await whisperer.explain_legacy_behavior(
        behavior=behavior, context=context
    )

    # Display results
    print(f"\n✓ Behavior Analysis Complete!")
    print(f"  Confidence: {explanation.confidence:.2%}")
    print(f"  Completeness: {explanation.completeness:.2%}")

    print(f"\n  Technical Explanation:")
    print(f"    {explanation.technical_explanation[:200]}...")

    print(f"\n  Root Causes: {len(explanation.root_causes)}")
    for cause in explanation.root_causes[:3]:
        print(f"    - {cause}")

    print(f"\n  Modernization Approaches: {len(explanation.modernization_approaches)}")
    for i, approach in enumerate(explanation.modernization_approaches[:3], 1):
        print(f"    {i}. {approach.get('name', 'Unknown')}")
        print(f"       Complexity: {approach.get('complexity', 'unknown')}")
        print(f"       Timeline: {approach.get('timeline', 'unknown')}")

    print(f"\n  Recommended Approach: {explanation.recommended_approach}")
    print(f"  Overall Risk Level: {explanation.risk_level.value}")

    print(f"\n  Modernization Risks: {len(explanation.modernization_risks)}")
    for risk in explanation.modernization_risks[:3]:
        print(
            f"    - {risk.get('category', 'unknown')}: {risk.get('description', '')[:60]}..."
        )

    print(f"\n  Implementation Steps: {len(explanation.implementation_steps)}")
    for i, step in enumerate(explanation.implementation_steps[:5], 1):
        print(f"    {i}. {step}")

    print(f"\n  Estimated Effort: {explanation.estimated_effort}")

    await whisperer.shutdown()


async def example_complete_workflow():
    """Example: Complete modernization workflow."""
    print("\n" + "=" * 60)
    print("Example 4: Complete Modernization Workflow")
    print("=" * 60)

    whisperer = await create_legacy_whisperer()

    # Step 1: Analyze legacy protocol
    print("\n[Step 1/4] Analyzing legacy protocol...")
    traffic_samples = [
        b"CMD:001|DATA:test1|END\n",
        b"CMD:002|DATA:test2|END\n",
        b"CMD:003|DATA:test3|END\n",
    ] * 5  # 15 samples

    spec = await whisperer.reverse_engineer_protocol(
        traffic_samples=traffic_samples, system_context="Legacy command protocol"
    )
    print(
        f"  ✓ Protocol analyzed: {spec.protocol_name} (confidence: {spec.confidence_score:.2%})"
    )

    # Step 2: Generate adapter
    print("\n[Step 2/4] Generating REST adapter...")
    adapter = await whisperer.generate_adapter_code(
        legacy_protocol=spec, target_protocol="REST", language=AdapterLanguage.PYTHON
    )
    print(f"  ✓ Adapter generated (quality: {adapter.code_quality_score:.2%})")

    # Step 3: Explain behavior
    print("\n[Step 3/4] Analyzing modernization options...")
    explanation = await whisperer.explain_legacy_behavior(
        behavior=f"Legacy protocol: {spec.protocol_name}",
        context={"protocol_spec": spec.spec_id},
    )
    print(f"  ✓ Analysis complete (risk: {explanation.risk_level.value})")

    # Step 4: Summary
    print("\n[Step 4/4] Modernization Summary")
    print(f"  Protocol: {spec.protocol_name}")
    print(f"  Complexity: {spec.complexity.value}")
    print(f"  Adapter Language: {adapter.language.value}")
    print(f"  Recommended Approach: {explanation.recommended_approach}")
    print(f"  Estimated Effort: {explanation.estimated_effort}")
    print(f"  Risk Level: {explanation.risk_level.value}")

    print(f"\n✓ Complete workflow finished successfully!")

    # Get statistics
    stats = whisperer.get_statistics()
    print(f"\n  Statistics:")
    print(f"    - Cached Specifications: {stats['analysis_cache_size']}")
    print(f"    - Cached Adapters: {stats['adapter_cache_size']}")

    await whisperer.shutdown()


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CRONOS AI - Legacy System Whisperer Examples")
    print("=" * 60)

    try:
        # Example 1: Reverse Engineering
        spec = await example_reverse_engineering()

        # Example 2: Adapter Generation
        await example_adapter_generation(spec)

        # Example 3: Behavior Explanation
        await example_behavior_explanation()

        # Example 4: Complete Workflow
        await example_complete_workflow()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
