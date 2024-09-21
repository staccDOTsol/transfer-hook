import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { TransferHook } from "../target/types/transfer_hook";
import {
  PublicKey,
  SystemProgram,
  Transaction,
  sendAndConfirmTransaction,
  Keypair,
  ComputeBudgetProgram,
  SYSVAR_RENT_PUBKEY,
} from "@solana/web3.js";
import {
  ExtensionType,
  TOKEN_2022_PROGRAM_ID,
  getMintLen,
  createInitializeMintInstruction,
  createInitializeTransferHookInstruction,
  ASSOCIATED_TOKEN_PROGRAM_ID,
  createAssociatedTokenAccountInstruction,
  createMintToInstruction,
  getAssociatedTokenAddressSync,
  createApproveInstruction,
  createSyncNativeInstruction,
  NATIVE_MINT,
  TOKEN_PROGRAM_ID,
  getAccount,
  getOrCreateAssociatedTokenAccount,
  createTransferCheckedWithTransferHookInstruction,
  getMint,
  getTransferHook,
  getExtraAccountMetaAddress,
  getExtraAccountMetas,
} from "@solana/spl-token";
import assert from "assert";
import { buildDefaultAccountFetcher, SwapUtils,buildWhirlpoolClient, increaseLiquidityQuoteByInputToken, ORCA_WHIRLPOOL_PROGRAM_ID, PDAUtil, PriceMath, TickArrayUtil, TickUtil, TokenExtensionUtil, Whirlpool, WhirlpoolAccountFetcher,WhirlpoolContext } from "../whirlpools/legacy-sdk/whirlpool/src";
import { BN } from "bn.js";
import { Percentage } from "@orca-so/common-sdk";
import Decimal from "decimal.js";
import { base64 } from "@coral-xyz/anchor/dist/cjs/utils/bytes";


function getTickArrayPublicKeysWithShift(
  tickCurrentIndex: number,
  tickSpacing: number,
  aToB: boolean,
  programId: PublicKey,
  whirlpoolAddress: PublicKey
) {
  let offset = 0;
  let tickArrayAddresses: {publicKey: PublicKey, index: number}[] = [];
  for (let i = 0; i < 100; i++) {
    let startIndex: number;
    try {
      const shift = aToB ? 0 : tickSpacing;
      startIndex = TickUtil.getStartTickIndex(tickCurrentIndex + shift, tickSpacing, offset);
    } catch {
      return tickArrayAddresses;
    }

    const pda = PDAUtil.getTickArray(programId, whirlpoolAddress, startIndex);
    tickArrayAddresses.push({publicKey: pda.publicKey, index: startIndex});
    offset = aToB ? offset - 1 : offset + 1;
  }

  return tickArrayAddresses;
}

describe("transfer-hook", () => {
  // Configure the client to use the local cluster.
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.TransferHook as Program<TransferHook>;
  const wallet = provider.wallet as anchor.Wallet;
  const connection = provider.connection;

  // Generate keypair to use as address for the transfer-hook enabled mint
  let mintKp = Keypair.generate();
  let otherMintKp = Keypair.generate();
  let mint = mintKp.publicKey;
  let otherMint = otherMintKp.publicKey;
  const decimals = 9;
const [positionAuthority] = PublicKey.findProgramAddressSync([Buffer.from("authority")], program.programId);
  // Sender token account address

let randomToken;
let randomWhirlpoolPair;
  // Create the two WSol token accounts as part of setup
let ctx:WhirlpoolContext
let pool : Whirlpool
let fetcher : WhirlpoolAccountFetcher
let gameData: any
  before(async () => {
    const extensions = [ExtensionType.TransferHook];
    const mintLen = getMintLen(extensions);
    const lamports =
      await provider.connection.getMinimumBalanceForRentExemption(mintLen);

    // @ts-ignore
    fetcher =  buildDefaultAccountFetcher(connection);
    // @ts-ignore
    ctx = WhirlpoolContext.withProvider(provider, ORCA_WHIRLPOOL_PROGRAM_ID, fetcher);
    // Create two mints
    // Select a random token from the top tokens list
    const topTokens = await (await fetch('https://cache.jup.ag/top-tokens')).json() as any;
    randomToken = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"//topTokens[Math.floor(Math.random() * topTokens.length)];
    console.log("Random token selected:", randomToken);

    // Select the pair of NATIVE_MINT vs something in top tokens from Orca whirlpool with highest TVL
    const whirlpools = await (await fetch('https://api.mainnet.orca.so/v1/whirlpool/list')).json() as any;
    const nativePairs = whirlpools.whirlpools.filter(pool => 
      (pool.tokenA.mint === NATIVE_MINT.toBase58() && pool.tokenB.mint === randomToken) || (pool.tokenB.mint === NATIVE_MINT.toBase58() && pool.tokenA.mint === randomToken)
    ).sort((a, b) => b.tvl - a.tvl);
    randomWhirlpoolPair = nativePairs[4];
    console.log("Highest TVL Orca whirlpool pair:", randomWhirlpoolPair);
    const client = buildWhirlpoolClient(ctx);

     pool = await client.getPool(new PublicKey(randomWhirlpoolPair.address));
     let [gameAccount] = PublicKey.findProgramAddressSync([Buffer.from("game"), pool.getAddress().toBuffer()], program.programId);
     try {
      gameData = await program.account.game.fetch(gameAccount);
     } catch (e) {
      console.log("Error fetching game data", e);
     }
     const mintA = pool.getTokenAInfo().mint;
     const mintB = pool.getTokenBInfo().mint;
     console.log("mintA", mintA.toBase58());
     console.log("mintB", mintB.toBase58());
    // Get account infos for all token accounts
    const accountInfos = await Promise.all([
      connection.getAccountInfo(getAssociatedTokenAddressSync(mintA, wallet.publicKey, true)),
      connection.getAccountInfo(getAssociatedTokenAddressSync(mintB, wallet.publicKey, true)),
      connection.getAccountInfo(getAssociatedTokenAddressSync(mintA, positionAuthority, true)),
      connection.getAccountInfo(getAssociatedTokenAddressSync(mintB, positionAuthority, true)),
      connection.getAccountInfo(getAssociatedTokenAddressSync(gameData.mint, wallet.publicKey, true, TOKEN_2022_PROGRAM_ID)),
      connection.getAccountInfo(getAssociatedTokenAddressSync(gameData.otherMint, wallet.publicKey, true, TOKEN_2022_PROGRAM_ID))
    ]);

    // Create instructions for accounts that don't exist
    const createAtaInstructions = accountInfos.map((info, index) => {
      if (info === null) {
        const m = index === 0 || index === 2 ? pool.getTokenAInfo().mint :
                  index === 1 || index === 3 ? pool.getTokenBInfo().mint :
                  index === 4 ? gameData.mint : gameData.otherMint;
        
        if (!m) return null;

        let owner = index < 2  || index > 3 ? wallet.publicKey : positionAuthority;

        const programId = index < 4 ? TOKEN_PROGRAM_ID : TOKEN_2022_PROGRAM_ID;
        return createAssociatedTokenAccountInstruction(
          wallet.publicKey,
          getAssociatedTokenAddressSync(m, owner, true, programId),
          owner,
          m,
          programId
        );
      }
      return null;
    }).filter(instruction => instruction !== null);

    // If there are any instructions to create accounts, send the transaction
    if (createAtaInstructions.length > 0) {
      const transaction = new Transaction().add(
        ComputeBudgetProgram.setComputeUnitPrice({microLamports: 20000}),
        ComputeBudgetProgram.setComputeUnitLimit({units: 1_400_000}),
        ...createAtaInstructions
      );
      const signature = await sendAndConfirmTransaction(connection, transaction, [wallet.payer]);
      console.log(`Created ${createAtaInstructions.length} associated token accounts. Signature: ${signature}`);
    }
  });

  // Account to store extra accounts required by the transfer hook instruction
  it("Create ExtraAccountMetaList Account", async () => {
    const positionMintKeypair = Keypair.generate();
    const positionPda = PDAUtil.getPosition(ORCA_WHIRLPOOL_PROGRAM_ID, positionMintKeypair.publicKey);
    const positionTokenAccountAddress = await getAssociatedTokenAddressSync(
      positionMintKeypair.publicKey,
      positionAuthority,
      true,
    );
    const mintA = pool.getTokenAInfo().mint 
    const mintB = pool.getTokenBInfo().mint 
    const currentTickIndex = pool.getData().tickCurrentIndex;
    const tickArrayAddresses = getTickArrayPublicKeysWithShift(currentTickIndex, pool.getData().tickSpacing, pool.getTokenAInfo().mint.equals(NATIVE_MINT), ORCA_WHIRLPOOL_PROGRAM_ID, pool.getAddress());
    const tickArrayAddresses2 = getTickArrayPublicKeysWithShift(currentTickIndex, pool.getData().tickSpacing, pool.getTokenBInfo().mint.equals(NATIVE_MINT), ORCA_WHIRLPOOL_PROGRAM_ID, pool.getAddress());
    const tickArrayLower = [...tickArrayAddresses, ...tickArrayAddresses2].find(tickArray => tickArray.index <= pool.getData().tickCurrentIndex);
    const tickArrayUpper = [...tickArrayAddresses, ...tickArrayAddresses2].find(tickArray => tickArray.index > pool.getData().tickCurrentIndex);
    console.log("tickArrayLower", tickArrayLower.publicKey.toBase58());
    console.log("tickArrayUpper", tickArrayUpper.publicKey.toBase58());
   
    let initializeExtraAccountMetaListInstruction = await program.methods
      .initializeFirstExtraAccountMetaList()
      .accounts({
        payer: wallet.publicKey,
        mint: mint,
        otherMint: otherMint,
        mintA: mintA,
        whirlpool: pool.getAddress(),
        tokenOwnerAccountA: getAssociatedTokenAddressSync(mintA, positionAuthority, true),
        tokenVaultA: pool.getTokenVaultAInfo().address,
        tokenOwnerAccountB: getAssociatedTokenAddressSync(mintB, positionAuthority, true),
        tokenVaultB: pool.getTokenVaultBInfo().address,
        oracle: PDAUtil.getOracle(ORCA_WHIRLPOOL_PROGRAM_ID, pool.getAddress()).publicKey,
      })
      .instruction();
    let transaction = new Transaction().add(
      ComputeBudgetProgram.setComputeUnitPrice({microLamports: 20000}),
      ComputeBudgetProgram.setComputeUnitLimit({units: 1_400_000}),
      initializeExtraAccountMetaListInstruction,
      await program.methods
      .initializeSecondExtraAccountMetaList()
      .accounts({
        payer: wallet.publicKey,
        mint: mint,
        mintB: mintB,
        otherMint: otherMint,
        whirlpool: pool.getAddress(),
        tokenOwnerAccountA: getAssociatedTokenAddressSync(mintA, positionAuthority, true),
        tokenVaultA: pool.getTokenVaultAInfo().address,
        tokenOwnerAccountB: getAssociatedTokenAddressSync(mintB, positionAuthority, true),
        tokenVaultB: pool.getTokenVaultBInfo().address,
        oracle: PDAUtil.getOracle(ORCA_WHIRLPOOL_PROGRAM_ID, pool.getAddress()).publicKey,
      })
      .instruction()
    );
    transaction.recentBlockhash = (await provider.connection.getLatestBlockhash()).blockhash;
    transaction.feePayer = wallet.publicKey;
    transaction.sign(wallet.payer, mintKp, otherMintKp);

    let txSig = await connection.sendRawTransaction(transaction.serialize());
    console.log("Transaction Signature:", txSig);
   
  const increase_quote = increaseLiquidityQuoteByInputToken(
    pool.getTokenBInfo().mint,
    new Decimal(await (await connection.getTokenAccountBalance(getAssociatedTokenAddressSync(pool.getTokenBInfo().mint, wallet.publicKey, true))).value.uiAmount / 100),
    tickArrayLower.index,
    tickArrayUpper.index,
    // @ts-ignore
    Percentage.fromFraction(1, 100),
    pool,
  await  TokenExtensionUtil.buildTokenExtensionContext(ctx.fetcher, pool.getData())
  );
  const minnttt = Math.random() < 0.5 ? gameData.mint : gameData.otherMint;
  const otherMinttt = minnttt === gameData.mint ? gameData.otherMint : gameData.mint;
console.log(positionPda.bump, tickArrayLower.index, tickArrayUpper.index, increase_quote.liquidityAmount)
 
const tx = await program.methods.openPosition(
  positionPda.bump,  tickArrayLower.index, tickArrayUpper.index, increase_quote.tokenMaxA, increase_quote.tokenMaxB
).accounts({
  owner: positionAuthority,
  tokenPositionAccountA: getAssociatedTokenAddressSync(mintA, positionAuthority, true),
  tokenPositionAccountB: getAssociatedTokenAddressSync(mintB, positionAuthority, true),
  funder: wallet.publicKey,
  otherMint: otherMinttt,
  userTokenAccountOther: getAssociatedTokenAddressSync(otherMinttt, wallet.publicKey, true, TOKEN_2022_PROGRAM_ID),
  userTokenAccountMint: getAssociatedTokenAddressSync(minnttt, wallet.publicKey, true, TOKEN_2022_PROGRAM_ID),
  mint: minnttt,
  tokenOwnerAccountA: getAssociatedTokenAddressSync(mintA, wallet.publicKey, true),
  tokenOwnerAccountB: getAssociatedTokenAddressSync(mintB, wallet.publicKey, true),
  tokenVaultA: pool.getTokenVaultAInfo().address,
  tokenVaultB: pool.getTokenVaultBInfo().address,
  tickArrayLower: tickArrayLower.publicKey,
  tickArrayUpper: tickArrayUpper.publicKey,
  positionMint: positionMintKeypair.publicKey,
  whirlpool: pool.getAddress(),
  position: positionPda.publicKey,
  positionTokenAccount: positionTokenAccountAddress,
  oracle: PDAUtil.getOracle(ORCA_WHIRLPOOL_PROGRAM_ID, pool.getAddress()).publicKey,
  
  }).
  preInstructions([ComputeBudgetProgram.setComputeUnitPrice({microLamports: 20000})])
  .signers([positionMintKeypair])
  .instruction();
   transaction = new Transaction().add(
    ComputeBudgetProgram.setComputeUnitPrice({microLamports: 20000}),
    ComputeBudgetProgram.setComputeUnitLimit({units: 1_400_000}),
    tx
  );
  transaction.recentBlockhash = (await provider.connection.getLatestBlockhash()).blockhash;
  transaction.feePayer = wallet.publicKey;
  transaction.sign(positionMintKeypair, wallet.payer);

   txSig = await connection.sendRawTransaction(transaction.serialize());
  console.log("created position 0 Transaction Signature:", txSig);

  });
  

  it("does a dozen proxyOpenPOsition with the transfer hook", async () => {
    const client = buildWhirlpoolClient(ctx);
    const pool = await client.getPool(new PublicKey(randomWhirlpoolPair.address));
    const mintA = pool.getTokenAInfo().mint;
    const mintB = pool.getTokenBInfo().mint;
    const whirlpool_data = pool.getData();
    const tickSpacing = whirlpool_data.tickSpacing;
    for (let i = 0; i < 12; i++) {
      const doneLowerTicks: number[] = [];
      const doneUpperTicks: number[] = [];
      console.log("current_tick_index", whirlpool_data.tickCurrentIndex);

      const tickArrays = await SwapUtils.getBatchTickArrays(
        ORCA_WHIRLPOOL_PROGRAM_ID,ctx.fetcher,
        [{tickCurrentIndex: whirlpool_data.tickCurrentIndex, tickSpacing: whirlpool_data.tickSpacing, aToB: true, whirlpoolAddress: pool.getAddress()},
        {tickCurrentIndex: whirlpool_data.tickCurrentIndex, tickSpacing: whirlpool_data.tickSpacing, aToB: false, whirlpoolAddress: pool.getAddress()}
        ],{maxAge: Number.MAX_SAFE_INTEGER}
      );
      console.log("tickArrays", tickArrays);
      
      // Find the closest lower and upper initialized ticks
      let tickLowerIndex = null;
      let tickUpperIndex = null;

      for (let j = 0; j < tickArrays.length; j++) {
        const tickArray = tickArrays[j];
        if (tickArray) {
          for (let k = 0; k < tickArray.length; k++) {
            const tickArrayData = tickArray[k];
            const startTickIndex = tickArrayData.startTickIndex;
            const endTickIndex = startTickIndex + (tickSpacing * 88); // 88 is the number of ticks in a tick array
            
            if (startTickIndex <= whirlpool_data.tickCurrentIndex && whirlpool_data.tickCurrentIndex < endTickIndex) {
              const lowerTick = Math.max(startTickIndex, whirlpool_data.tickCurrentIndex - (4 * tickSpacing));
              const upperTick = Math.min(endTickIndex - 1, whirlpool_data.tickCurrentIndex + (4 * tickSpacing));
              
              console.log("lowerTick", lowerTick);
              console.log("upperTick", upperTick);
              
                tickLowerIndex = lowerTick;
                tickUpperIndex = upperTick;
              
              
              if (tickLowerIndex !== null && tickUpperIndex !== null) {
                break;
              }
            }
          }
        }
        if (tickLowerIndex !== null && tickUpperIndex !== null) {
          break;
        }
      }
      const positionMintKeypair = Keypair.generate();
      const positionPda = PDAUtil.getPosition(ORCA_WHIRLPOOL_PROGRAM_ID, positionMintKeypair.publicKey);
      const positionTokenAccountAddress = await getAssociatedTokenAddressSync(
        positionMintKeypair.publicKey,
        positionAuthority,
        true,
      );
      const mintA = pool.getTokenAInfo().mint 
      const mintB = pool.getTokenBInfo().mint 
      const currentTickIndex = pool.getData().tickCurrentIndex;
      const tickArrayAddresses = getTickArrayPublicKeysWithShift(currentTickIndex, pool.getData().tickSpacing, pool.getTokenAInfo().mint.equals(NATIVE_MINT), ORCA_WHIRLPOOL_PROGRAM_ID, pool.getAddress());
      const tickArrayAddresses2 = getTickArrayPublicKeysWithShift(currentTickIndex, pool.getData().tickSpacing, pool.getTokenBInfo().mint.equals(NATIVE_MINT), ORCA_WHIRLPOOL_PROGRAM_ID, pool.getAddress());
      const tickArrayLower = [...tickArrayAddresses, ...tickArrayAddresses2].find(tickArray => tickArray.index <= pool.getData().tickCurrentIndex);
      const tickArrayUpper = [...tickArrayAddresses, ...tickArrayAddresses2].find(tickArray => tickArray.index > pool.getData().tickCurrentIndex);
      console.log("tickArrayLower", tickArrayLower.publicKey.toBase58());
      console.log("tickArrayUpper", tickArrayUpper.publicKey.toBase58());
      tickLowerIndex =  tickArrayLower.index;
      tickUpperIndex = tickArrayUpper.index;
      console.log("tickLowerIndex", tickLowerIndex);
      console.log("tickUpperIndex", tickUpperIndex);

      const increase_quote = increaseLiquidityQuoteByInputToken(
        pool.getTokenBInfo().mint,
        new Decimal(await (await connection.getTokenAccountBalance(getAssociatedTokenAddressSync(pool.getTokenBInfo().mint, wallet.publicKey, true))).value.uiAmount / 100),
        tickLowerIndex,
        tickUpperIndex,
        // @ts-ignore
        Percentage.fromFraction(1, 100),
        pool,
           await  TokenExtensionUtil.buildTokenExtensionContext(ctx.fetcher, pool.getData())

      );
     
      const minnttt = Math.random() < 0.5 ? gameData.mint : gameData.otherMint;
      const otherMinttt = minnttt === gameData.mint ? gameData.otherMint : gameData.mint;
    console.log(positionPda.bump, tickLowerIndex, tickUpperIndex, increase_quote.liquidityAmount)
    try {
     
    const tx = await program.methods.openPosition(
      positionPda.bump,  tickLowerIndex, tickUpperIndex, increase_quote.tokenMaxA, increase_quote.tokenMaxB
    ).accounts({
      owner: positionAuthority,
      tokenPositionAccountA: getAssociatedTokenAddressSync(mintA, positionAuthority, true),
      tokenPositionAccountB: getAssociatedTokenAddressSync(mintB, positionAuthority, true),
      funder: wallet.publicKey,
      otherMint: otherMinttt,
      userTokenAccountOther: getAssociatedTokenAddressSync(otherMinttt, wallet.publicKey, true, TOKEN_2022_PROGRAM_ID),
      userTokenAccountMint: getAssociatedTokenAddressSync(minnttt, wallet.publicKey, true, TOKEN_2022_PROGRAM_ID),
      mint: minnttt,
      tokenOwnerAccountA: getAssociatedTokenAddressSync(mintA, wallet.publicKey, true),
      tokenOwnerAccountB: getAssociatedTokenAddressSync(mintB, wallet.publicKey, true),
      tokenVaultA: pool.getTokenVaultAInfo().address,
      tokenVaultB: pool.getTokenVaultBInfo().address,
      tickArrayLower: tickArrayLower.publicKey,
      tickArrayUpper: tickArrayUpper.publicKey,
      positionMint: positionMintKeypair.publicKey,
      whirlpool: pool.getAddress(),
      position: positionPda.publicKey,
      positionTokenAccount: positionTokenAccountAddress,
      oracle: PDAUtil.getOracle(ORCA_WHIRLPOOL_PROGRAM_ID, pool.getAddress()).publicKey,
      
    }).
    preInstructions([ComputeBudgetProgram.setComputeUnitPrice({microLamports: 20000})])
    .signers([positionMintKeypair])
    .instruction();
    const transaction = new Transaction().add(

      ComputeBudgetProgram.setComputeUnitPrice({microLamports: 20000}),
      ComputeBudgetProgram.setComputeUnitLimit({units: 1_400_000}),
      tx
    );
    transaction.recentBlockhash = (await provider.connection.getLatestBlockhash()).blockhash;
    transaction.feePayer = wallet.publicKey;
    transaction.sign(positionMintKeypair, wallet.payer);
    const txSig = await connection.sendRawTransaction(transaction.serialize());
    console.log("created position", i, "Transaction Signature:", txSig);

  } catch (e) {
    console.log("Error creating position", i, e);
  }

  }
  })
  it("Transfer Hook with Extra Account Meta", async () => {
    // 1 tokens
    const amount = 1;
    const bigIntAmount = BigInt(amount);

    for (let i = 0; i < 1999999990; i++) {
      await new Promise(resolve => setTimeout(resolve, Math.random() * 10000+1000));
    // Standard token transfer instruction
    const receipient = Keypair.generate().publicKey;
    
    var transferInstruction = await createTransferCheckedWithTransferHookInstruction(
      connection,
      getAssociatedTokenAddressSync(gameData.mint, wallet.publicKey, true, TOKEN_2022_PROGRAM_ID),
      gameData.mint,
      getAssociatedTokenAddressSync(gameData.mint, receipient, true, TOKEN_2022_PROGRAM_ID),
      wallet.publicKey,
      bigIntAmount,
      pool.getTokenAInfo().decimals,
      [],
      "confirmed",
      TOKEN_2022_PROGRAM_ID
    );


    var transaction = new Transaction().add(
      ComputeBudgetProgram.setComputeUnitPrice({microLamports: 20000}),
      ComputeBudgetProgram.setComputeUnitLimit({units: 1_400_000}),
      createAssociatedTokenAccountInstruction(wallet.publicKey, getAssociatedTokenAddressSync(gameData.mint, receipient, true, TOKEN_2022_PROGRAM_ID), receipient, gameData.mint, TOKEN_2022_PROGRAM_ID),
      transferInstruction
    );
    transaction.recentBlockhash = (await provider.connection.getLatestBlockhash()).blockhash;
    transaction.feePayer = wallet.publicKey;
    transaction.sign(wallet.payer);
    var txSig = await connection.sendRawTransaction(transaction.serialize());
    console.log("Transfer Signature:", txSig);

    var transferInstruction = await createTransferCheckedWithTransferHookInstruction(
      connection,
      getAssociatedTokenAddressSync(gameData.otherMint, wallet.publicKey, true, TOKEN_2022_PROGRAM_ID),
     gameData.otherMint,
      getAssociatedTokenAddressSync(gameData.otherMint, receipient, true, TOKEN_2022_PROGRAM_ID),
      wallet.publicKey,
      bigIntAmount,
      pool.getTokenBInfo().decimals,
      [],
      "confirmed",
      TOKEN_2022_PROGRAM_ID
    );


    var transaction = new Transaction().add(
      ComputeBudgetProgram.setComputeUnitPrice({microLamports: 20000}),
      ComputeBudgetProgram.setComputeUnitLimit({units: 1_400_000}),
      createAssociatedTokenAccountInstruction(wallet.publicKey, getAssociatedTokenAddressSync(gameData.otherMint, receipient, true, TOKEN_2022_PROGRAM_ID), receipient, gameData.otherMint, TOKEN_2022_PROGRAM_ID),
      transferInstruction
    );
    transaction.recentBlockhash = (await provider.connection.getLatestBlockhash()).blockhash;
    transaction.feePayer = wallet.publicKey;
    transaction.sign(wallet.payer);
    var txSig = await connection.sendRawTransaction(transaction.serialize());
    console.log("Transfer Signature:", txSig);
  }

  });
});
